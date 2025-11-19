library(tidyverse)
library(arrow)
library(here)
library(progressr)  # For progress bars
library(future.apply)  # For parallel processing
library(data.table)  # For faster data operations

# Enable parallel processing
plan(multisession)

# Load data
cat("Loading data...\n")
load(here("data", "X_data"))
results <- list(
  forecasts    = read_parquet(here("output", "forecasts.parquet")),
  active_coefs = read_parquet(here("output", "active_coefs.parquet")),
  meta         = jsonlite::read_json(here("output", "meta.json"), simplifyVector = TRUE)
)

# Data prep
cat("Preparing data...\n")
X <- X_data %>% arrange(date)
topic_cols <- str_subset(names(X), "^Topic_")
stock_cols <- setdiff(names(X), c("date", topic_cols))
K <- 3
L <- 30

# Optimized lag creation using data.table for speed
create_lags_optimized <- function(data, cols, K, suffix) {
  dt <- as.data.table(data)
  
  # Pre-allocate result list
  lag_list <- vector("list", length(cols) * K)
  idx <- 1
  
  for (col in cols) {
    for (k in 0:(K-1)) {
      lag_list[[idx]] <- shift(dt[[col]], k, type = "lag")
      names(lag_list)[idx] <- paste0(col, "_", suffix, "Lag", k)
      idx <- idx + 1
    }
  }
  
  as_tibble(lag_list)
}

cat("Creating lagged variables...\n")
Xlag <- bind_cols(
  X %>% select(date),
  create_lags_optimized(X, stock_cols, K, "R"),
  create_lags_optimized(X, topic_cols, K, "T")
) %>%
  slice(K:n()) # Remove rows without complete lags

# Vectorized AR(1) estimation function
compute_persistence_vectorized <- function(x) {
  # Remove leading/trailing NAs
  valid_idx <- which(!is.na(x))
  if (length(valid_idx) < 2) {
    return(tibble(phi = NA_real_, p_value = NA_real_, significant = FALSE))
  }
  
  x_clean <- x[valid_idx]
  n <- length(x_clean)
  
  if (n < 2) {
    return(tibble(phi = NA_real_, p_value = NA_real_, significant = FALSE))
  }
  
  y <- x_clean[-1]
  y_lag <- x_clean[-n]
  
  # Fast linear regression using matrix operations
  n_obs <- length(y)
  if (n_obs < 2) {
    return(tibble(phi = NA_real_, p_value = NA_real_, significant = FALSE))
  }
  
  X_mat <- cbind(1, y_lag)
  
  tryCatch({
    # Solve normal equations directly
    XtX_inv <- solve(crossprod(X_mat))
    coefs <- XtX_inv %*% crossprod(X_mat, y)
    
    # Calculate residuals and standard errors
    residuals <- y - X_mat %*% coefs
    sigma_sq <- sum(residuals^2) / (n_obs - 2)
    se <- sqrt(diag(XtX_inv) * sigma_sq)
    
    phi <- coefs[2, 1]
    t_stat <- phi / se[2]
    p_val <- 2 * pt(-abs(t_stat), df = n_obs - 2)
    
    tibble(
      phi = phi,
      p_value = p_val,
      significant = is.finite(p_val) && p_val < 0.05
    )
  }, error = function(e) {
    tibble(phi = NA_real_, p_value = NA_real_, significant = FALSE)
  })
}






# Analysis 1: Persistence of actively selected predictors

# Pre-process active coefficients data
active_coefs_processed <- results$active_coefs %>%
  mutate(date_t = as.Date(date_t)) %>%
  left_join(
    Xlag %>% 
      mutate(date_pos = row_number()) %>% 
      select(date, date_pos),
    by = c("date_t" = "date")
  ) %>%
  filter(!is.na(date_pos)) %>%
  mutate(
    window_start_pos = pmax(1, date_pos - L),
    window_end_pos = date_pos - 1
  ) %>%
  filter(window_end_pos >= window_start_pos)

# Processing with detailed progress bar
n_active_estimations <- nrow(active_coefs_processed)

# Convert to data.table for faster operations
dt_active <- as.data.table(active_coefs_processed)
dt_xlag <- as.data.table(Xlag)

# Process each row efficiently with progress tracking
persistence_results <- vector("list", n_active_estimations)

# Create progress bar that shows counts
pb <- txtProgressBar(min = 0, max = n_active_estimations, style = 3)

for (i in seq_len(n_active_estimations)) {
  # Update progress bar with count information
  setTxtProgressBar(pb, i)
  if (i %% 1000 == 0 || i == n_active_estimations) {
    cat(sprintf("\rActive predictors: %d / %d estimations completed (%.1f%%)", 
                i, n_active_estimations, (i / n_active_estimations) * 100))
    flush.console()
  }
  
  row_data <- dt_active[i]
  pred_name <- as.character(row_data$predictor)
  start_pos <- row_data$window_start_pos
  end_pos <- row_data$window_end_pos
  
  if (start_pos >= 1 && end_pos >= start_pos && pred_name %in% names(dt_xlag)) {
    # Extract series data efficiently using data.table
    series_data <- dt_xlag[start_pos:end_pos, ..pred_name][[1]]
    persistence_results[[i]] <- compute_persistence_vectorized(series_data)
  } else {
    persistence_results[[i]] <- tibble(phi = NA_real_, p_value = NA_real_, significant = FALSE)
  }
}

close(pb)

# Combine results
active_persistence <- active_coefs_processed %>%
  bind_cols(bind_rows(persistence_results)) %>%
  mutate(
    window_start = Xlag$date[pmax(1, window_start_pos)],
    window_end = Xlag$date[pmax(1, window_end_pos)],
    phi_hat = ifelse(significant, phi, NA_real_)
  ) %>%
  select(target_stock, date_t, predictor, window_start, window_end, 
         phi_hat, p_value, significant)

'active_persistence_unique <- active_persistence %>%
  distinct(date_t, predictor, .keep_all = TRUE)'

# Plot for active predictors
daily_active <- active_persistence %>%
  group_by(date_t) %>%
  summarise(share_true = mean(significant, na.rm = TRUE), .groups = "drop")

share_stats <- daily_active %>%
  summarise(
    n = n(),
    m = mean(share_true, na.rm = TRUE),
    sd = sd(share_true, na.rm = TRUE),
    median = median(share_true, na.rm = TRUE),
    min_val = min(share_true, na.rm = TRUE),
    max_val = max(share_true, na.rm = TRUE),
    q25 = quantile(share_true, 0.25, na.rm = TRUE),
    q75 = quantile(share_true, 0.75, na.rm = TRUE)
  )

# Set y-axis limits with some padding
y_limits <- c(
  max(0, share_stats$min_val - 0.01), 
  share_stats$max_val + 0.01
)

# Create the time series plot
p_daily_share_ts <- ggplot(daily_active, aes(x = date_t, y = share_true)) +
  geom_line(color = "darkred", alpha = 0.8, size = 0.6) +
  geom_hline(yintercept = share_stats$m, linewidth = 0.6, color = "steelblue") +
  annotate("label", x = max(daily_active$date_t), y = share_stats$m,
           label = paste0("Mean: ", percent(share_stats$m, accuracy = 0.1)),
           hjust = 1, vjust = -0.4, size = 3, color = "steelblue") +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = y_limits) +
  labs(
    x = "Date", 
    y = "Daily Share",
    title = "Daily Share of Persistent Predictors",
    subtitle = paste0(
      "N = ", scales::comma(share_stats$n),
      " | Mean = ", percent(share_stats$m, accuracy = 0.1),
      " | SD = ", percent(share_stats$sd, accuracy = 0.1),
      " | Median = ", percent(share_stats$median, accuracy = 0.1),
      " | Range = ", percent(share_stats$min_val, accuracy = 0.1),
      " - ", percent(share_stats$max_val, accuracy = 0.1)
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 30),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Display the time series plot
print(p_daily_share_ts)

##################################################################################
write_parquet(active_persistence, file.path(here("output"), "persistence_active.parquet"))
ggsave(
  filename = "output/daily_persistence_plot.png",  # file path
  plot = p_daily_share_ts,                                   # plot object
  width = 8, height = 5, dpi = 300            # dimensions & resolution
)
##################################################################################