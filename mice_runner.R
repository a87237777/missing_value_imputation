#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: Rscript mice_runner.R <in_csv> <out_prefix> <m> <maxit> <seed> <method>")
}
in_csv    <- args[[1]]
out_pref  <- args[[2]]
m         <- as.integer(args[[3]])
maxit     <- as.integer(args[[4]])
seed      <- as.integer(args[[5]])
method    <- args[[6]]

suppressPackageStartupMessages({
  library(mice)
})

df <- read.csv(in_csv, check.names = FALSE, stringsAsFactors = FALSE, na.strings=c("NA",""))

# Convert numeric-like character columns back to numeric
for (nm in names(df)) {
  x <- df[[nm]]
  if (is.character(x)) {
    suppressWarnings({
      x_num <- as.numeric(x)
      nn <- sum(!is.na(x))
      ok <- (nn > 0) && (sum(!is.na(x_num)) >= max(1, floor(0.95 * nn)))
      if (ok) df[[nm]] <- x_num
    })
  }
}

set.seed(seed)

imp <- suppressMessages(
  suppressWarnings(
    mice(df, m=m, maxit=maxit, method=method, printFlag=FALSE)
  )
)

for (k in 1:m) {
  comp <- complete(imp, action=k)
  out_path <- paste0(out_pref, "_", k, ".csv")
  write.csv(comp, out_path, row.names = FALSE, na = "")
}
