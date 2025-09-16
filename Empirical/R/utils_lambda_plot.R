plot_lambda_vs_predictors <- function(out, title = "Lambda vs Number of Active Predictors") {
  # Load required library
  library(ggplot2)
  
  # Extract the forecasts data
  forecasts_data <- out$forecasts
  
  # Filter to focus on cases where predictors are active (> 0) and lambda <= 0.03
  active_data <- forecasts_data[forecasts_data$n_active > 0 & forecasts_data$lambda <= 0.03, ]
  
  # Create the plot
  p <- ggplot(active_data, aes(x = lambda, y = n_active)) +
    geom_point(alpha = 0.6, color = "steelblue") +
    scale_x_continuous(breaks = seq(0, 0.035, by = 0.001), 
                       limits = c(0, 0.03)) +
    labs(
      title = title,
      x = "Lambda Value",
      y = "Number of Active Predictors",
      caption = paste("Observations with active predictors (λ ≤ 0.03):", nrow(active_data), 
                     "out of", nrow(forecasts_data), "total")
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  # Add trend line without confidence interval if there's variation in both variables
  if (length(unique(active_data$lambda)) > 1 && 
      length(unique(active_data$n_active)) > 1) {
    p <- p + geom_smooth(method = "loess", se = FALSE, color = "red")
  }
