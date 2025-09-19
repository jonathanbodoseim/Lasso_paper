suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

# --- R2_oos ---

X_long <- X_data %>%
  mutate(week = as.Date(week)) %>%                    # ensure Date for join
  tidyr::pivot_longer(-week, names_to = "target_stock", values_to = "realized")

df <- forecasts %>%
  mutate(week_pred = as.Date(week_pred)) %>%          # ensure Date for join
  left_join(X_long, by = c("week_pred" = "week", "target_stock"))

oos_r2 <- df %>%
  group_by(target_stock) %>%
  summarise(
    mse_model = mean((realized - forecast)^2, na.rm = TRUE),
    mse_naive = mean((realized - mean(realized, na.rm = TRUE))^2, na.rm = TRUE),
    r2_oos = 1 - mse_model / mse_naive,
    .groups = "drop"
  )

# --- Plot active predictors vs lambda ---

ggplot(results$forecasts, aes(x = lambda, y = n_predictors)) +
  geom_point(alpha = 0.6, size = 1.5) +
  labs(
    title = "Lasso Regularization: Lambda vs Number of Active Predictors",
    x = "Lambda (Regularization Parameter)",
    y = "Number of Active Predictors",
    subtitle = "Each point represents one forecast across all stocks and time periods"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50")
  ) +
  scale_x_continuous(trans = "log10", labels = scales::scientific_format()) +
  annotation_logticks(sides = "b")
