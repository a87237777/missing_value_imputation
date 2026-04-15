#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript missforest_runner.R <in_csv> <out_csv> <maxiter> <seed>")
}
in_csv  <- args[[1]]
out_csv <- args[[2]]
maxiter <- as.integer(args[[3]])
seed    <- as.integer(args[[4]])

suppressPackageStartupMessages({
  library(missForest)
})

df <- read.csv(in_csv, check.names = FALSE, stringsAsFactors = FALSE, na.strings=c("NA",""))

# Convert obvious numeric-like character columns back to numeric
for (nm in names(df)) {
  x <- df[[nm]]
  if (is.character(x)) {
    suppressWarnings({
      x_num <- as.numeric(x)
      # if most non-missing parse, use numeric
      nn <- sum(!is.na(x))
      ok <- (nn > 0) && (sum(!is.na(x_num)) >= max(1, floor(0.95 * nn)))
      if (ok) df[[nm]] <- x_num
    })
  }
}

set.seed(seed)
fit <- missForest(as.matrix(df), maxiter = maxiter, verbose = FALSE)
ximp <- fit$ximp

write.csv(ximp, out_csv, row.names = FALSE, na = "")
