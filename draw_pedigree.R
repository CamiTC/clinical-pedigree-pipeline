library(Pedixplorer)

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]                        # path to input/CSV file
output_png <- args[2]                       # path to save output

df <- read.csv(input_csv, stringsAsFactors = FALSE, na.strings = c("", "NA"))

# change sex to integer: 1 = male, 2 = female, 3 = unknown
df$sex <- ifelse(df$sex == "M", 1, ifelse(df$sex == "F", 2, 3))

# change affected to integer: 1 = affected, 0 = unaffected
df$affected <- as.integer(df$affected)

# convert deceased to True or False
df$status <- as.logical(df$status)

# split multi-line label into name and conditions
df$name <- sapply(df$label, function(x) {
  strsplit(as.character(x), "\n")[[1]][1]
})
df$cond <- sapply(df$label, function(x) {
  lines <- strsplit(as.character(x), "\n")[[1]]
  if (length(lines) > 1) paste(lines[-1], collapse = "\n") else ""
})

# hide phantom node labels
is_phantom <- grepl("^__ph", df$id)
df$name[is_phantom] <- ""
df$cond[is_phantom] <- ""

# constructs Pedixplorer Pedigree object from df
ped <- Pedigree(df)

# set affected fill
fill_colour <- fill(scales(ped))
fill_colour$fill[fill_colour$affected == TRUE] <- "black"
fill_colour$fill[fill_colour$affected == FALSE] <- "white"
fill(scales(ped)) <- fill_colour

# open PNG graphics device
png(output_png, width = 2000, height = 1200, res = 150, bg = "transparent")
par(mar = c(4, 4, 4, 4))

# draw pedigree and capture coordinates for arrow placement
ped_coords <- plot(ped, id_lab = "name", label = "cond",
                   cex = 0.55, aff_mark = FALSE, width = 6)

# restore coordinate system; required after plot() resets par(usr)
par(usr = ped_coords$par_usr$usr, xpd = TRUE)

# find proband symbol in plot data
p0_sym <- ped_coords$df[
  grepl("<b>Proband</b>", ped_coords$df$tips, fixed = TRUE) &
    grepl("(square|circle)", ped_coords$df$type),
]

# get proband coordinates (centre)
if (nrow(p0_sym) > 0) {
  px <- p0_sym$x0[1]
  py <- p0_sym$y0[1]
  boxw <- ped_coords$par_usr$boxw
  boxh <- ped_coords$par_usr$boxh

  # draw proband arrow
  arrows(
    x0 = px - boxw * 2.0,
    y0 = py + boxh * 2.0,
    x1 = px - boxw * 0.65,
    y1 = py + boxh * 0.65,
    length = 0.08, lwd = 2, col = "black"
  )
}

dev.off()

cat("Saved:", output_png, "\n")
