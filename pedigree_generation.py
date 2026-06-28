import subprocess           # calls R from python
import pandas as pd
from pathlib import Path

# stores the path to draw_pedigree.R
R_SCRIPT = Path(__file__).parent / "draw_pedigree.R"

# used to infer sex
FEMALES = {"mother", "daughter", "sister", "aunt", "grandmother", "niece", "wife"}
MALES   = {"father", "son", "brother", "uncle", "grandfather", "nephew", "husband"}


# ----------------
# helper functions
# ----------------

def _sex(relative, sex=None):
    """
    Checks the person's sex from the "sex" field or uses the relative label.
    Returns: 'M', 'F', or 'U'.
    """
    # check sex field first
    if sex and str(sex).lower() in ("male", "m"): return "M"
    if sex and str(sex).lower() in ("female", "f"): return "F"
    
    rel = str(relative).lower()
    
    # otherwise infer sex from relative label
    if rel in MALES: return "M"
    if rel in FEMALES: return "F"
    
    # unknown
    return "U"


def _affected(person_df):
    """
    Assigns affected status.
    Returns: 1 if the person has at least one present condition, otherwise 0 if none.
    """
    for _, row in person_df.iterrows():
        cond = str(row.get("condition", "")).lower().strip()
        polarity = str(row.get("polarity",  "")).lower().strip()
        
        if cond not in ("none", "", "nan") and polarity == "present":
            return 1
        
    return 0


def _clean_age(age):
    """
    Strips the .0 from float ages. 
    Returns: the cleaned age or None if missing.
    """
    if pd.isna(age): return None
    age_str = str(age).strip()
    return age_str[:-2] if age_str.endswith(".0") else age_str


def _format_age(age, unit):
    """
    Adds the age unit abbreviation to the age.
    Returns: formatted age.
    """
    if not age: return age

    unit = str(unit).strip().lower() if pd.notna(unit) else ""

    if unit == "years":  return f"{age} y.o."
    if unit == "months": return f"{age} m.o."
    if unit == "weeks": return f"{age} w.o."
    if unit == "days": return f"{age} d.o."

    return f"{age} {unit}".strip()


def _label(person_df):
    """
    Creates the label for each person with name, age, conditions with diagnosis age.
    Returns: complete label.
    """
    first_row = person_df.iloc[0]
    parts = [str(first_row["relative"]).title()]

    # add current age
    age = _clean_age(first_row.get("current_age"))
    
    if age:
        parts.append(_format_age(age, first_row.get("current_age_unit", "")))

    # tracks conditions that are already added
    seen = set()

    for _, row in person_df.iterrows():
        raw_cond = str(row.get("condition", "")).lower().strip()
        polarity = str(row.get("polarity", "")).lower().strip()

        # skip missing or absent conditions
        if raw_cond in ("none", "", "nan") or polarity == "absent" or raw_cond in seen:
            continue

        seen.add(raw_cond)
        cond = raw_cond.capitalize()
        suffix = "?" if polarity == "uncertain" else ""

        # add age of diagnosis / event
        event_age = _clean_age(row.get("age_at_event"))

        if event_age:
            parts.append(f"{cond}{suffix} (dx. {_format_age(event_age, row.get('age_at_event_unit', ''))})")
        else:
            parts.append(f"{cond}{suffix}")

    # add deceased age
    if str(person_df.iloc[0].get("deceased", "")).lower().strip() == "true":
        for _, row in person_df.iterrows():
            raw_cond = str(row.get("condition", "")).lower().strip()
            event_age = _clean_age(row.get("age_at_event"))
            
            if event_age and raw_cond in ("none", "", "nan"):
                parts.append(f"d. {_format_age(event_age, row.get('age_at_event_unit', ''))}")
                break

    return "\n".join(parts)     # join all parts of label


def _classify(people):
    """
    Classifies every person based on their family role.
    """
    roles = {
        "mother": None,
        "father": None,
        "maternal_gm": None,
        "maternal_gf": None,
        "paternal_gm": None,
        "paternal_gf": None,
        "maternal_au": [],
        "paternal_au": [],
        "siblings": [],
        "children": [],
    }
    for person_id, person_df in people.items():
        rel = str(person_df.iloc[0]["relative"]).lower()
        side = str(person_df.iloc[0].get("side", "")).lower()

        if rel == "mother": roles["mother"] = person_id

        elif rel == "father": roles["father"] = person_id

        elif rel == "grandmother":
            if side == "maternal": roles["maternal_gm"] = person_id
            else: 
                roles["paternal_gm"] = person_id  # paternal or unknown -> paternal
      
        elif rel == "grandfather":
            if side == "maternal": roles["maternal_gf"] = person_id
            else:
                roles["paternal_gf"] = person_id  # paternal or unknown -> paternal

        elif rel in ("aunt", "uncle"):
            if side == "maternal": roles["maternal_au"].append(person_id)
            else:
                roles["paternal_au"].append(person_id)  # paternal or unknown → paternal
        
        elif rel in ("brother", "sister"): roles["siblings"].append(person_id)
        
        elif rel in ("son", "daughter"):   roles["children"].append(person_id)
    
    return roles


def _to_pedigree_df(people, roles):
    """
    Converts people dict to a Pedixplorer-compatible DataFrame.
    Adds phantom parent nodes to properly connect children.
    """
    parent_map = {}         # maps person ID to (father ID, mother ID)
    phantom_rows = []       # stores phantom node rows
    phantom_n = [0]         # list so nested function can change it         

    def new_phantom(sex):
        """
        Creates a new unlabelled parent node required by Pedixplorer.
        """

        phantom_n[0] += 1
        person_id = f"__ph{phantom_n[0]}"
        phantom_rows.append({
            "id": person_id, "dadid": "", "momid": "",
            "sex": sex, "affected": 0, "status": 0, "label": "",
        })
        return person_id

    def fill_parents(gf_id, gm_id):
        """
        If one parent is known, create a phantom for the missing one.
        """
        if gf_id is not None and gm_id is None:
            gm_id = new_phantom("F")

        elif gm_id is not None and gf_id is None:
            gf_id = new_phantom("M")

        return gf_id, gm_id

    def ensure_parents(group, gf_id, gm_id):
        """
        Creates phantom parents if a child group needs to be drawn.
        Also fills in any missing single parent with a phantom.
        Returns the gf_id, gm_id.
        """
        if gf_id is None and gm_id is None:
            if len(group) > 1:                  # multiple siblings with no parents
                gf_id = new_phantom("M")
                gm_id = new_phantom("F")
        else:
            gf_id, gm_id = fill_parents(gf_id, gm_id)

        return gf_id, gm_id

    # maternal side: mother + maternal aunts/uncles
    mat_side = ([roles["mother"]] if roles["mother"] else []) + roles["maternal_au"]

    if mat_side:
        maternal_gf, maternal_gm = ensure_parents(mat_side, roles["maternal_gf"], roles["maternal_gm"])
        
        for person_id in mat_side:
            parent_map[person_id] = (maternal_gf, maternal_gm)          # connect grandparents
    
    # no maternal children, grandparents stay as root nodes
    else:
        maternal_gf, maternal_gm = roles["maternal_gf"], roles["maternal_gm"]

    # paternal side: father + paternal aunts/uncles
    pat_side = ([roles["father"]] if roles["father"] else []) + roles["paternal_au"]
    
    if pat_side:
        paternal_gf, paternal_gm = ensure_parents(pat_side, roles["paternal_gf"], roles["paternal_gm"])
        
        for person_id in pat_side:
            parent_map[person_id] = (paternal_gf, paternal_gm)
    
    else:
        paternal_gf, paternal_gm = roles["paternal_gf"], roles["paternal_gm"]

    # grandparents are root nodes
    for gp in (maternal_gf, maternal_gm, paternal_gf, paternal_gm):
        if gp and not gp.startswith("__ph") and gp not in parent_map:
            parent_map[gp] = (None, None)

    # proband + siblings share the same parents
    p0_group = ["P0"] + roles["siblings"]
    p0_father, p0_mother = ensure_parents(p0_group, roles["father"], roles["mother"])
    
    for person_id in p0_group:
        parent_map[person_id] = (p0_father, p0_mother)

    # proband's children: assign P0 as dad or mom based on sex
    p0_row = people["P0"].iloc[0] if "P0" in people else {}      # empty dict if proband is missing
    
    # if sex is unknown, default P0 to male
    p0_sex = _sex(p0_row.get("relative", ""), p0_row.get("sex")) if "P0" in people else "U"
    
    if roles["children"]:
       
        # female proband
        if p0_sex == "F":
            ph_spouse = new_phantom("M")
            
            for person_id in roles["children"]:
                parent_map[person_id] = (ph_spouse, "P0")
        
        # male proband
        else:
            ph_spouse = new_phantom("F")
            
            for person_id in roles["children"]:
                parent_map[person_id] = ("P0", ph_spouse)

    # build row for each real person
    rows = []

    for person_id, person_df in people.items():
        dadid, momid = parent_map.get(person_id, (None, None))
        row = person_df.iloc[0]
        sex = _sex(row.get("relative", ""), row.get("sex"))
        
        # if P0 sex is still unknown but has children, default to M
        if person_id == "P0" and sex == "U" and roles["children"]:
            sex = "M"
        
        rows.append({
            "id": person_id,
            "dadid": dadid if dadid else "",
            "momid": momid if momid else "",
            "sex": sex,
            "affected": _affected(person_df),
            "status": 1 if str(person_df.iloc[0].get("deceased", "")).lower().strip() == "true" else 0,
            "label": _label(person_df),
        })

    return pd.DataFrame(rows + phantom_rows)


# -------------
# main function
# -------------

def draw_pedigree(df, output_dir, output_name):
    """
    Draws and saves a pedigree chart using Pedixplorer
    """
    people = dict(tuple(df.groupby("person_id", sort=False)))   # group records by person ID
    roles = _classify(people)
    kin_df = _to_pedigree_df(people, roles)     # build DataFrame for Pedixplorer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{output_name}.csv"
    png_path = out_dir / f"{output_name}.png"

    kin_df.to_csv(csv_path, index=False)

    # call R script
    result = subprocess.run(
        ["Rscript", str(R_SCRIPT), str(csv_path), str(png_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  R error for {output_name}:\n{result.stderr}")
        return None

    return png_path
