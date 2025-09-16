# scripts/_load_functions.R
suppressPackageStartupMessages({
  library(here)
  library(purrr)
})

# Function to source all R files in R/ folder
source_dir <- function(dir_path) {
  files <- list.files(dir_path, pattern = "\\.[Rr]$", full.names = TRUE)
  purrr::walk(files, ~ sys.source(.x, envir = globalenv()))
}

# Load everything from R/
source_dir(here("Empirical", "R"))