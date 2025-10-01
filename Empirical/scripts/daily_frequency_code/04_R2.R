# ---- Setup ----
library(here)
library(arrow)      # for read_parquet
library(dplyr)
library(tidyr)
library(stringr)
library(broom)      # for glance()
library(ggplot2)
library(scales)
library(purrr) 

# ---- Load data and prepr ----
load(here("data", "X_data"))  # -> X_data (has 'date' + stock columns)
forecasts_lasso   <- arrow::read_parquet(here("output", "forecasts.parquet"))
forecasts_ar3 <- arrow::read_parquet(here("output", "forecasts_ar3.parquet"))


forecasts_ar3_wide <- na.omit(forecasts_ar3)
forecasts_lasso_wide <- forecasts_lasso %>%
  select(target_stock, date_t, forecast) %>%
  pivot_wider(
    names_from = target_stock,
    values_from = forecast,
    names_prefix = "stock_"
  ) %>%
  arrange(date_t)

# Function to standardize forecasts (z-score) - only for numeric columns
standardize_forecasts <- function(forecasts) {
  date_col <- names(forecasts)[grepl("date", names(forecasts), ignore.case = TRUE)]
  
  forecasts %>%
    mutate(across(-all_of(date_col), ~ {
      if(is.numeric(.x)) {
        (.x - mean(.x, na.rm = TRUE)) / sd(.x, na.rm = TRUE)
      } else {
        .x
      }
    }))
}

# Standardize both forecast datasets
forecasts_lasso_std <- standardize_forecasts(forecasts_lasso_wide)
forecasts_ar3_std <- standardize_forecasts(forecasts_ar3_wide)

# Get common stocks and dates
common_stocks <- intersect(
  sub("^stock_", "", setdiff(names(forecasts_lasso_std), "date_t")),
  setdiff(names(forecasts_ar3_std), "date_t")
)

lasso_common <- forecasts_lasso_std %>%
  mutate(date_t = as.POSIXct(date_t)) %>%
  rename_with(~ sub("^stock_", "", .x), starts_with("stock_")) %>%
  select(date_t, all_of(common_stocks))

ar3_common <- forecasts_ar3_std %>%
  mutate(date_t = as.POSIXct(date_t)) %>%  # already POSIXct, harmless
  select(date_t, all_of(common_stocks))

# Remove "stock_" prefix from LASSO forecasts 
names(forecasts_lasso_std) <- gsub("^stock_", "", names(forecasts_lasso_std))
common_stocks <- gsub("^stock_", "", common_stocks)

# Prepare realized returns data
returns_data <- X_data %>%
  rename(date_t = date) %>%
  select(date_t, all_of(common_stocks))

# Function to run regression for a single stock
run_stock_regression <- function(stock_id, returns_df, forecasts_df, model_name) {
  
  if (!stock_id %in% names(returns_df) || !stock_id %in% names(forecasts_df)) {
    return(tibble(
      stock = stock_id,
      model = model_name,
      alpha = NA_real_, beta = NA_real_, r_squared = NA_real_, 
      p_value_alpha = NA_real_, p_value_beta = NA_real_,
      n_obs = 0L
    ))
  }
  
  # Merge returns and forecasts
  merged_data <- returns_df %>%
    select(date_t, all_of(stock_id)) %>%
    rename(return = all_of(stock_id)) %>%
    inner_join(
      forecasts_df %>% 
        select(date_t, all_of(stock_id)) %>%
        rename(forecast = all_of(stock_id)),
      by = "date_t"
    ) %>%
    filter(!is.na(return), !is.na(forecast))
  
  if (nrow(merged_data) < 10) {  # Minimum observations requirement
    return(tibble(
      stock = stock_id,
      model = model_name,
      alpha = NA_real_, beta = NA_real_, r_squared = NA_real_, 
      p_value_alpha = NA_real_, p_value_beta = NA_real_,
      n_obs = as.integer(nrow(merged_data))
    ))
  }
  
  # Run regression: r_i,t = alpha_i + beta_i * z_i,t + epsilon_i,t
  tryCatch({
    reg_model <- lm(return ~ forecast, data = merged_data)
    reg_summary <- summary(reg_model)
    
    tibble(
      stock = stock_id,
      model = model_name,
      alpha = as.numeric(coef(reg_model)[1]),
      beta = as.numeric(coef(reg_model)[2]),
      r_squared = as.numeric(reg_summary$r.squared),
      p_value_alpha = as.numeric(reg_summary$coefficients[1, 4]),
      p_value_beta = as.numeric(reg_summary$coefficients[2, 4]),
      n_obs = as.integer(nrow(merged_data))
    )
  }, error = function(e) {
    tibble(
      stock = stock_id,
      model = model_name,
      alpha = NA_real_, beta = NA_real_, r_squared = NA_real_, 
      p_value_alpha = NA_real_, p_value_beta = NA_real_,
      n_obs = as.integer(nrow(merged_data))
    )
  })
}

# Function to run combined regression (both LASSO and AR3 forecasts)
run_combined_regression <- function(stock_id, returns_df, lasso_df, ar3_df) {
  
  # Check if stock exists in all datasets
  if (!stock_id %in% names(returns_df) || !stock_id %in% names(lasso_df) || !stock_id %in% names(ar3_df)) {
    return(tibble(
      stock = stock_id,
      model = "combined",
      alpha = NA_real_, beta_lasso = NA_real_, beta_ar3 = NA_real_, r_squared = NA_real_,
      p_value_alpha = NA_real_, p_value_beta_lasso = NA_real_, p_value_beta_ar3 = NA_real_,
      n_obs = 0L
    ))
  }
  
  # Merge all data
  merged_data <- returns_df %>%
    select(date_t, all_of(stock_id)) %>%
    rename(return = all_of(stock_id)) %>%
    inner_join(
      lasso_df %>% 
        select(date_t, all_of(stock_id)) %>%
        rename(forecast_lasso = all_of(stock_id)),
      by = "date_t"
    ) %>%
    inner_join(
      ar3_df %>% 
        select(date_t, all_of(stock_id)) %>%
        rename(forecast_ar3 = all_of(stock_id)),
      by = "date_t"
    ) %>%
    filter(!is.na(return), !is.na(forecast_lasso), !is.na(forecast_ar3))
  
  if (nrow(merged_data) < 10) {
    return(tibble(
      stock = stock_id,
      model = "combined",
      alpha = NA_real_, beta_lasso = NA_real_, beta_ar3 = NA_real_, r_squared = NA_real_,
      p_value_alpha = NA_real_, p_value_beta_lasso = NA_real_, p_value_beta_ar3 = NA_real_,
      n_obs = as.integer(nrow(merged_data))
    ))
  }
  
  # Run combined regression: r_i,t = alpha_i + beta_lasso * z_lasso_i,t + beta_ar3 * z_ar3_i,t + epsilon_i,t
  tryCatch({
    reg_model <- lm(return ~ forecast_lasso + forecast_ar3, data = merged_data)
    reg_summary <- summary(reg_model)
    
    tibble(
      stock = stock_id,
      model = "combined",
      alpha = as.numeric(coef(reg_model)[1]),
      beta_lasso = as.numeric(coef(reg_model)[2]),
      beta_ar3 = as.numeric(coef(reg_model)[3]),
      r_squared = as.numeric(reg_summary$r.squared),
      p_value_alpha = as.numeric(reg_summary$coefficients[1, 4]),
      p_value_beta_lasso = as.numeric(reg_summary$coefficients[2, 4]),
      p_value_beta_ar3 = as.numeric(reg_summary$coefficients[3, 4]),
      n_obs = as.integer(nrow(merged_data))
    )
  }, error = function(e) {
    tibble(
      stock = stock_id,
      model = "combined",
      alpha = NA_real_, beta_lasso = NA_real_, beta_ar3 = NA_real_, r_squared = NA_real_,
      p_value_alpha = NA_real_, p_value_beta_lasso = NA_real_, p_value_beta_ar3 = NA_real_,
      n_obs = as.integer(nrow(merged_data))
    )
  })
}

# Run regressions for all stocks
lasso_results <- map_dfr(common_stocks, ~run_stock_regression(.x, returns_data, forecasts_lasso_std, "lasso"))
ar3_results <- map_dfr(common_stocks, ~run_stock_regression(.x, returns_data, forecasts_ar3_std, "ar3"))
combined_results <- map_dfr(common_stocks, ~run_combined_regression(.x, returns_data, forecasts_lasso_std, forecasts_ar3_std))


# Calculate statistics for each model
calculate_stats <- function(data, r2_col) {
  list(
    n = nrow(data),
    m = mean(data[[r2_col]], na.rm = TRUE),
    sd = sd(data[[r2_col]], na.rm = TRUE),
    median = median(data[[r2_col]], na.rm = TRUE),
    n_pos = sum(data[[r2_col]] > 0, na.rm = TRUE),
    share_pos = mean(data[[r2_col]] > 0, na.rm = TRUE)
  )
}

# Statistics for each model
lasso_stats <- calculate_stats(lasso_results, "r_squared")
ar3_stats <- calculate_stats(ar3_results, "r_squared")
combined_stats <- calculate_stats(combined_results, "r_squared")

# Create Lasso R² plot
p_lasso_r2 <- ggplot(lasso_results, aes(x = "", y = r_squared)) +
  geom_jitter(width = 0.08, height = 0, color = "darkred", alpha = 0.6, size = 1.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_hline(yintercept = lasso_stats$m, linewidth = 0.6, color = "steelblue") +
  annotate("label", x = 1.03, y = lasso_stats$m,
           label = paste0("Mean: ", percent(lasso_stats$m, accuracy = 0.1)),
           hjust = 0, vjust = -0.4, size = 3, color = "steelblue") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    x = NULL, y = expression(R^2),
    title = "Per-stock Lasso R²",
    subtitle = paste0(
      "N = ", scales::comma(lasso_stats$n),
      " | Mean = ", percent(lasso_stats$m, accuracy = 0.1),
      " | SD = ", percent(lasso_stats$sd, accuracy = 0.1),
      " | Median = ", percent(lasso_stats$median, accuracy = 0.1),
      " | Positive = ", lasso_stats$n_pos,
      " | Share > 0 = ", percent(lasso_stats$share_pos, accuracy = 0.1)
    )
  ) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 30)
  )

# Create AR3 R² plot
p_ar3_r2 <- ggplot(ar3_results, aes(x = "", y = r_squared)) +
  geom_jitter(width = 0.08, height = 0, color = "darkred", alpha = 0.6, size = 1.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_hline(yintercept = ar3_stats$m, linewidth = 0.6, color = "steelblue") +
  annotate("label", x = 1.03, y = ar3_stats$m,
           label = paste0("Mean: ", percent(ar3_stats$m, accuracy = 0.1)),
           hjust = 0, vjust = -0.4, size = 3, color = "steelblue") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    x = NULL, y = expression(R^2),
    title = "Per-stock AR3 R²",
    subtitle = paste0(
      "N = ", scales::comma(ar3_stats$n),
      " | Mean = ", percent(ar3_stats$m, accuracy = 0.1),
      " | SD = ", percent(ar3_stats$sd, accuracy = 0.1),
      " | Median = ", percent(ar3_stats$median, accuracy = 0.1),
      " | Positive = ", ar3_stats$n_pos,
      " | Share > 0 = ", percent(ar3_stats$share_pos, accuracy = 0.1)
    )
  ) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 30)
  )

# Create Combined R² plot
p_combined_r2 <- ggplot(combined_results, aes(x = "", y = r_squared)) +
  geom_jitter(width = 0.08, height = 0, color = "darkred", alpha = 0.6, size = 1.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_hline(yintercept = combined_stats$m, linewidth = 0.6, color = "steelblue") +
  annotate("label", x = 1.03, y = combined_stats$m,
           label = paste0("Mean: ", percent(combined_stats$m, accuracy = 0.1)),
           hjust = 0, vjust = -0.4, size = 3, color = "steelblue") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    x = NULL, y = expression(R^2),
    title = "Per-stock Combined R²",
    subtitle = paste0(
      "N = ", scales::comma(combined_stats$n),
      " | Mean = ", percent(combined_stats$m, accuracy = 0.1),
      " | SD = ", percent(combined_stats$sd, accuracy = 0.1),
      " | Median = ", percent(combined_stats$median, accuracy = 0.1),
      " | Positive = ", combined_stats$n_pos,
      " | Share > 0 = ", percent(combined_stats$share_pos, accuracy = 0.1)
    )
  ) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 30)
  )

# Create delta R² plot (Combined vs AR3)
# Merge the datasets to calculate the difference
delta_combined_ar3 <- merge(combined_results[c("stock", "r_squared")], 
                            ar3_results[c("stock", "r_squared")], 
                            by = "stock", suffixes = c("_combined", "_ar3"))
delta_combined_ar3$delta_r2 <- delta_combined_ar3$r_squared_combined - delta_combined_ar3$r_squared_ar3

# Calculate statistics for delta
delta_stats <- calculate_stats(delta_combined_ar3, "delta_r2")

# Calculate symmetric limits around 0 for centering
delta_range <- range(delta_combined_ar3$delta_r2, na.rm = TRUE)
max_abs_delta <- max(abs(delta_range))
y_limits <- c(-max_abs_delta, max_abs_delta)

p_delta_combined_ar3_r2 <- ggplot(delta_combined_ar3, aes(x = "", y = delta_r2)) +
  geom_jitter(width = 0.08, height = 0, color = "darkred", alpha = 0.6, size = 1.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_hline(yintercept = delta_stats$m, linewidth = 0.6, color = "steelblue") +
  annotate("label", x = 1.03, y = delta_stats$m,
           label = paste0("Mean: ", percent(delta_stats$m, accuracy = 0.1)),
           hjust = 0, vjust = -0.4, size = 3, color = "steelblue") +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = y_limits) +
  labs(
    x = NULL, y = expression(Delta~R^2~(Combined~-~AR3)),
    title = "Per-stock Δ R² (Combined vs AR3)",
    subtitle = paste0(
      "N = ", scales::comma(delta_stats$n),
      " | Mean = ", percent(delta_stats$m, accuracy = 0.1),
      " | SD = ", percent(delta_stats$sd, accuracy = 0.1),
      " | Median = ", percent(delta_stats$median, accuracy = 0.1),
      " | Positive = ", delta_stats$n_pos,
      " | Share > 0 = ", percent(delta_stats$share_pos, accuracy = 0.1)
    )
  ) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 30)
  )

# Save plots using ggsave
ggsave(
  here("output", "lasso_r2_plot.png"),  # full path + filename
  plot = p_lasso_r2,
  width = 8, height = 6, dpi = 300
)
ggsave(
  here("output", "ar3_r2_plot.png"),  # full path + filename
  plot = p_ar3_r2,
  width = 8, height = 6, dpi = 300
)
ggsave(
  here("output", "combined_r2_plot.png"),  # full path + filename
  plot = p_combined_r2,
  width = 8, height = 6, dpi = 300
)
ggsave(
  here("output", "delta_combined_ar3_r2_plot.png"),  # full path + filename
  plot = p_delta_combined_ar3_r2,
  width = 8, height = 6, dpi = 300
)

# Display the plots
print(p_lasso_r2)
print(p_ar3_r2)
print(p_combined_r2)
print(p_delta_combined_ar3_r2)