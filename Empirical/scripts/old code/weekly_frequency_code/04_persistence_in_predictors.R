suppressPackageStartupMessages({
  library(data.table)
})

# --- persistence analysis for selected predictors (simple AR(1) on rolling window) ---

# Helper: estimate a AR(1) model for a time series vector v
.compute_phi_p <- function(v) {
  x   <- v[-1]
  x_l <- v[-length(v)]
  fit <- lm(x ~ x_l)
  sm  <- summary(fit)
  list(phi = unname(coef(sm)[2, 1]), p = unname(coef(sm)[2, 4]))
}

# Build a fast week -> index lookup on the Xlag matrix timeline
week_to_pos <- setNames(seq_along(weeks_t), as.character(weeks_t))

# create an empty list to collect the persistence analysis results
persistence_rows <- list()

# extract table of predictors that were selected by the rolling LASSO model
# contains target_stock, week_t, predictor
ac <- results$active_coefs
ac[, week_t := as.Date(week_t)]  # ensure Date for lookup

# loop over all predictors selected by the rolling LASSO model (iterate over each row in ac)
for (i in seq_len(nrow(ac))) {
  tgt   <- ac$target_stock[i]              # which stock was predicted
  w_t   <- ac$week_t[i]                    # week of the forecast
  pred  <- ac$predictor[i]                 # which predictor was selected
  tpos  <- week_to_pos[as.character(w_t)]  # Find the position t on the Xlag/Weeks timeline
  if (is.na(tpos)) next
  lo <- tpos - L                           # Use the same rolling window that produced the forecast: [t-L, t-1]
  hi <- tpos - 1L
  v <- Xlag[[pred]][lo:hi]                 # extract return/topic vector within rolling window
  est <- .compute_phi_p(v)                 # use helper to estimate AR1 on window
  sig <- is.finite(est$p) && est$p < 0.05  # check if rho is significant (predictor is persistent)
  persistence_rows[[length(persistence_rows) + 1L]] <- data.frame(  # store results after each iteration
    target_stock = tgt,
    week_t       = w_t,
    predictor    = pred,
    window_start = weeks_t[lo],
    window_end   = weeks_t[hi],
    phi_hat      = if (sig) est$phi else NA_real_,
    p_value      = est$p,
    significant  = sig,
    stringsAsFactors = FALSE
  )
}

persistence_dt <- if (length(persistence_rows)) data.table::rbindlist(persistence_rows) else data.table()

# --- Save results -------------------------------------------------------------
out_path <- here("output", "persistence_results.csv")
fwrite(persistence_dt, out_path)




