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

# Build a fast date -> index lookup on the Xlag matrix timeline
date_to_pos <- setNames(seq_along(dates_t), as.character(dates_t))

# create an empty list to collect the persistence analysis results
persistence_rows <- list()

# extract table of predictors that were selected by the rolling LASSO model
# contains target_stock, date_t, predictor
ac <- results$active_coefs
ac[, date_t := as.Date(date_t)]  # ensure Date for lookup

# loop over all predictors selected by the rolling LASSO model (iterate over each row in ac)
for (i in seq_len(nrow(ac))) {
  tgt   <- ac$target_stock[i]              # which stock was predicted
  w_t   <- ac$date_t[i]                    # date of the forecast
  pred  <- ac$predictor[i]                 # which predictor was selected
  tpos  <- date_to_pos[as.character(w_t)]  # Find the position t on the Xlag/dates timeline
  if (is.na(tpos)) next
  lo <- tpos - L                           # Use the same rolling window that produced the forecast: [t-L, t-1]
  hi <- tpos - 1L
  v <- Xlag[[pred]][lo:hi]                 # extract return/topic vector within rolling window
  est <- .compute_phi_p(v)                 # use helper to estimate AR1 on window
  sig <- is.finite(est$p) && est$p < 0.05  # check if rho is significant (predictor is persistent)
  persistence_rows[[length(persistence_rows) + 1L]] <- data.frame(  # store results after each iteration
    target_stock = tgt,
    date_t       = w_t,
    predictor    = pred,
    window_start = dates_t[lo],
    window_end   = dates_t[hi],
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


dately_share <- persistence_dt[
  , .(share_true = mean(significant, na.rm = TRUE)),
  by = date_t
]

# --- Plot results -------------------------------------------------------------

p <-  ggplot(dately_share, aes(x = date_t, y = share_true)) +
  geom_line(color = "steelblue") +
  geom_point(color = "darkred", size = 0.8) +
  labs(
    title = "dately share of active predictors with ts persistence",
    x = "date",
    y = "%"
  ) +
  theme_minimal()

p

ggsave(
  filename = "output/dately_share_plot.png",  # file path
  plot = p,                                   # plot object
  width = 8, height = 5, dpi = 300            # dimensions & resolution
)

# --- persistence analysis for all predictors (simple AR(1) on rolling window) ---

suppressPackageStartupMessages({
  library(data.table)
})

# helper: estimate AR(1) phi and p-value (same as yours)
.compute_phi_p <- function(v) {
  if(length(v) < 3 || all(is.na(v))) return(list(phi=NA_real_, p=NA_real_))
  x   <- v[-1]
  x_l <- v[-length(v)]
  # drop NA pairs
  ok <- !(is.na(x) | is.na(x_l))
  if(sum(ok) < 3) return(list(phi=NA_real_, p=NA_real_))
  fit <- lm(x[ok] ~ x_l[ok])
  sm  <- summary(fit)
  # coef matrix might not have 2 rows if degenerate; guard it
  if(nrow(coef(sm)) < 2) return(list(phi=NA_real_, p=NA_real_))
  list(phi = unname(coef(sm)[2, 1]), p = unname(coef(sm)[2, 4]))
}

# compute persistence for ALL predictors in Xlag
compute_persistence_all <- function(Xlag, dates_t, L) {
  # normalize Xlag access: support data.frame / data.table / list / matrix
  if(is.data.frame(Xlag) || is.data.table(Xlag)) {
    preds <- setdiff(colnames(Xlag), c("date","date","date","Date"))
    get_series <- function(pred) Xlag[[pred]]
  } else if(is.list(Xlag) && !is.null(names(Xlag))) {
    preds <- names(Xlag)
    get_series <- function(pred) Xlag[[pred]]
  } else if(is.matrix(Xlag)) {
    preds <- colnames(Xlag)
    get_series <- function(pred) Xlag[, pred]
  } else stop("Unsupported Xlag type. Provide data.frame, list or matrix.")
  
  ndates <- length(dates_t)
  date_to_pos <- setNames(seq_along(dates_t), as.character(dates_t))
  rows <- vector("list", length = length(preds) * max(0, ndates - L))
  rpos <- 0L
  pb <- txtProgressBar(min=0, max=length(preds), style=3)
  for(i in seq_along(preds)) {
    pred <- preds[i]
    series <- get_series(pred)
    # ensure series length matches dates_t if vector-like
    if(length(series) < ndates) {
      # try recycling only if it is a list of full length; otherwise skip
      warning(sprintf("predictor '%s' shorter than dates_t: skipping", pred))
      next
    }
    for(tpos in seq_len(ndates)) {
      lo <- tpos - L
      hi <- tpos - 1L
      if(lo < 1L || hi < 1L) next
      v <- series[lo:hi]
      est <- .compute_phi_p(v)
      sig <- is.finite(est$p) && est$p < 0.05
      rpos <- rpos + 1L
      rows[[rpos]] <- list(
        predictor    = pred,
        date_t       = as.Date(dates_t[tpos]),
        window_start = as.Date(dates_t[lo]),
        window_end   = as.Date(dates_t[hi]),
        phi_hat      = if (sig) est$phi else NA_real_,
        p_value      = est$p,
        significant  = sig
      )
    }
    setTxtProgressBar(pb, i)
  }
  close(pb)
  if(rpos == 0L) return(data.table())
  persistence_dt <- rbindlist(rows[seq_len(rpos)], use.names = TRUE, fill = TRUE)
  setcolorder(persistence_dt, c("predictor","date_t","window_start","window_end","phi_hat","p_value","significant"))
  persistence_dt
}

# ----------Compute persistence and plot ----------
persistence_dt <- compute_persistence_all(Xlag = Xlag, dates_t = dates_t, L = 30)

frac_by_date <- persistence_dt[ , .(frac_sig = mean(significant, na.rm = TRUE),
                                    n_obs = .N), by = date_t] %>% arrange(date_t)

ggplot(frac_by_date, aes(x = date_t, y = frac_sig)) +
  geom_line(size = 0.9) +
  scale_y_continuous(labels = scales::percent_format(1)) +
  labs(title = "Fraction of predictors significant (AR(1) p<0.05)",
       x = "date", y = "fraction significant") +
  theme_minimal()
