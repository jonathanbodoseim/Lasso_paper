suppressPackageStartupMessages({
  library(data.table)
  library(glmnet)
  library(doParallel)
  library(readr)  
  library(here)  
  library(ggplot2)
  library(dplyr)
  library(arrow)
  library(jsonlite)
  library(scales)
})

# --- load data ---

load(here("data", "X_data"))
results <- list(
  forecasts    = read_parquet(file.path(here("output"), "forecasts.parquet")) |> as.data.table(),
  active_coefs = read_parquet(file.path(here("output"), "active_coefs.parquet")) |> as.data.table(),
  meta         = read_json(file.path(here("output"), "meta.json"), simplifyVector = TRUE)
)

# --- Plot active predictors vs lambda ---

n_points <- nrow(results$forecasts)

lambda_means <- results$forecasts %>%
  filter(n_predictors %in% c(5,10,15,20,25,30,35,40)) %>%
  group_by(n_predictors) %>%
  summarise(mean_lambda = mean(lambda, na.rm = TRUE), .groups = "drop")

# --- Plot ---
n_points <- sum(!is.na(results$forecasts$lambda) & !is.na(results$forecasts$n_predictors))
y_top    <- max(results$forecasts$n_predictors, na.rm = TRUE)
pad      <- diff(range(results$forecasts$n_predictors, na.rm = TRUE)) * 0.04

p <- ggplot(results$forecasts, aes(x = lambda, y = n_predictors)) +
  geom_point(color = "darkred", alpha = 0.6, size = 1.5) +
  # vertical dashed lines
  geom_vline(
    data = lambda_means,
    aes(xintercept = mean_lambda),
    linetype = "dashed", linewidth = 0.5, color = "steelblue"
  ) +
  # labels at the very top of each line
  geom_text(
    data = lambda_means,
    aes(x = mean_lambda, y = y_top + pad, label = n_predictors),
    vjust = 0, size = 3, color = "steelblue", inherit.aes = FALSE
  ) +
  labs(
    title = "Lasso Regularization: Lambda vs Number of Active Predictors",
    subtitle = paste0("Each point = one forecast | N = ", scales::comma(n_points)),
    x = "Lambda",
    y = "Number of Active Predictors"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 25)  # extra space on top for labels
  ) +
  scale_x_continuous(trans = "log10", labels = scientific_format()) +
  annotation_logticks(sides = "b")

##################################################################################
ggsave(
  filename = "output/lambdavspredictors.png",  # file path
  plot = p,                                   # plot object
  width = 8, height = 5, dpi = 300            # dimensions & resolution
)
##################################################################################
# Calculate daily average lambda values
daily_lambda <- forecasts_lasso %>%
  group_by(date_t) %>%
  summarise(
    avg_lambda = mean(lambda, na.rm = TRUE),
    .groups = 'drop'
  )

# Calculate summary statistics for daily average lambda
lambda_stats <- daily_lambda %>%
  summarise(
    n = n(),
    m = mean(avg_lambda, na.rm = TRUE),
    sd = sd(avg_lambda, na.rm = TRUE),
    median = median(avg_lambda, na.rm = TRUE),
    min_val = min(avg_lambda, na.rm = TRUE),
    max_val = max(avg_lambda, na.rm = TRUE),
    q25 = quantile(avg_lambda, 0.25, na.rm = TRUE),
    q75 = quantile(avg_lambda, 0.75, na.rm = TRUE)
  )

# Set y-axis limits with some padding
y_limits <- c(
  max(0, lambda_stats$min_val - 0.001), 
  lambda_stats$max_val + 0.001
)

# Create the time series plot for average lambda per day
p_daily_lambda <- ggplot(daily_lambda, aes(x = date_t, y = avg_lambda)) +
  geom_line(color = "darkred", alpha = 0.8, size = 0.6) +
  geom_hline(yintercept = lambda_stats$m, linewidth = 0.6, color = "steelblue") +
  annotate("label", x = max(daily_lambda$date_t), y = lambda_stats$m,
           label = paste0("Mean: ", round(lambda_stats$m, 4)),
           hjust = 1, vjust = -0.4, size = 3, color = "steelblue") +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
  scale_y_continuous(labels = function(x) round(x, 4), limits = y_limits) +
  labs(
    x = "Date", 
    y = "Average Lambda",
    title = "Average Daily Penalty Parameter",
    subtitle = paste0(
      "N = ", scales::comma(lambda_stats$n),
      " | Mean = ", round(lambda_stats$m, 4),
      " | SD = ", round(lambda_stats$sd, 4),
      " | Median = ", round(lambda_stats$median, 4),
      " | Range = ", round(lambda_stats$min_val, 4),
      " - ", round(lambda_stats$max_val, 4)
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 30),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

##################################################################################
print(p_daily_lambda)
ggsave(
  filename = "output/p_daily_lambda.png",  # file path
  plot = p_daily_lambda,                                   # plot object
  width = 8, height = 5, dpi = 300            # dimensions & resolution
)
##################################################################################


