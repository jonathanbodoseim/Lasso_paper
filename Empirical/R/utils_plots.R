# R/utils_plots.R

#' Plot heatmaps of active predictors per target (from run_lasso_ret_and_topics)
#'
#' @param out List returned by run_lasso_ret_and_topics().
#' @param permno_ticker data.frame/data.table with columns PERMNO, TICKER (character).
#' @param top_n Keep top-N predictors by frequency (across full history) per target.
#' @param save_dir If not NULL, saves one PNG per target into this directory.
#' @return A named list of ggplot objects (one per target); invisibly saves PNGs if save_dir provided.
#' @export
plot_active_predictors_per_target <- function(out, permno_ticker, top_n = 40, save_dir = NULL){
  # ---- input checks -----------------------------------------------------------
  if (is.null(out$active_coefs) || nrow(data.table::as.data.table(out$active_coefs)) == 0) {
    stop("No active_coefs found in 'out'. Re-run with nonzero_tol <= 0 and ensure there are active predictors.")
  }
  ac <- data.table::as.data.table(out$active_coefs)
  fc <- data.table::as.data.table(out$forecasts)[, .(target_stock)]
  targets <- unique(fc$target_stock)
  
  permno_ticker <- data.table::as.data.table(permno_ticker)
  if (!all(c("PERMNO","TICKER") %in% names(permno_ticker))) {
    stop("permno_ticker must have columns: PERMNO, TICKER")
  }
  permno_ticker[, `:=`(PERMNO = as.character(PERMNO), TICKER = as.character(TICKER))]
  
  # ---- per-target plot builder -----------------------------------------------
  make_one <- function(target_id){
    dt <- ac[target_stock == target_id]
    if (!nrow(dt)) return(NULL)
    
    # classify predictor type via suffix
    dt[, type := ifelse(grepl("_RLag\\d+$", predictor), "Return",
                        ifelse(grepl("_TLag\\d+$", predictor), "Topic", "Other"))]
    
    # parse numeric lag from predictor name; supports any K (0..K-1)
    dt[, lag_num := data.table::fifelse(type == "Return",
                                        as.integer(sub(".*_RLag(\\d+)$", "\\1", predictor)),
                                        as.integer(sub(".*_TLag(\\d+)$", "\\1", predictor)))]
    # text label like "Lag0", "Lag1", ... ordered by numeric
    lag_levels <- sort(unique(dt$lag_num))
    dt[, lag := factor(paste0("Lag", lag_num), levels = paste0("Lag", lag_levels))]
    
    # base name (PERMNO for returns; Topic_* for topics)
    dt[, base := data.table::fifelse(type == "Return",
                                     sub("_RLag\\d+$", "", predictor),
                                     sub("_TLag\\d+$", "", predictor))]
    
    # attach display label
    dt[type == "Return", TICKER := permno_ticker$TICKER[match(base, permno_ticker$PERMNO)]]
    dt[type == "Topic",  TICKER := base]
    dt[is.na(TICKER) & type == "Return", TICKER := paste0("PERMNO_", base)]
    
    # row label includes lag
    dt[, y_lab := paste0(TICKER, " (", lag, ")")]
    
    # keep top-N predictors by frequency over history (collapsing lags)
    freq <- dt[, .N, by = .(base)][order(-N)]
    keep_base <- head(freq$base, top_n)
    viz <- dt[base %in% keep_base]
    if (!nrow(viz)) return(NULL)
    
    # order rows: group by TICKER then by lag order; top to bottom
    viz[, grp := TICKER]
    row_levels <- viz[order(grp, lag), unique(y_lab)]
    viz[, y_lab := factor(y_lab, levels = rev(row_levels))]
    
    # coerce week_t to Date (IDate is fine too)
    if (!inherits(viz$week_t, "Date") && !inherits(viz$week_t, "IDate")) {
      suppressWarnings(viz[, week_t := data.table::as.IDate(week_t)])
    }
    
    # title with target's ticker if known
    target_ticker <- permno_ticker$TICKER[match(as.character(target_id), permno_ticker$PERMNO)]
    title_txt <- if (!is.na(target_ticker) && nzchar(target_ticker)) {
      sprintf("Active LASSO Predictors — target: %s (%s)", target_ticker, target_id)
    } else {
      sprintf("Active LASSO Predictors — target: %s", target_id)
    }
    
    p <- ggplot2::ggplot(viz, ggplot2::aes(x = week_t, y = y_lab, fill = lag)) +
      ggplot2::geom_tile(height = 0.9) +
      ggplot2::scale_y_discrete(name = "Predictor") +
      ggplot2::scale_x_date(name = "Week t (predicting t+1)",
                            date_breaks = "1 month", date_labels = "%Y-%m") +
      ggplot2::labs(title = title_txt, fill = "Lag") +
      ggplot2::theme_minimal(base_size = 14) +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
        panel.grid.minor = ggplot2::element_blank()
      )
    
    if (!is.null(save_dir)) {
      if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
      fn <- file.path(save_dir, sprintf("lasso_active_predictors_%s.png", target_id))
      ggplot2::ggsave(fn, p, width = 12, height = 8, dpi = 150)
    }
    
    p
  }
  
  plots <- stats::setNames(vector("list", length(targets)), targets)
  for (i in seq_along(targets)) {
    plots[[i]] <- make_one(targets[i])
  }
  plots
}
