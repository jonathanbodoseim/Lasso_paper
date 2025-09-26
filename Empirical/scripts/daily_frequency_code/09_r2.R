suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

# load data 
load(here("data", "X_data"))
forecasts_ar3     <- read_parquet(file.path(here("output"), "forecasts_ar3.parquet")) %>% as.data.table()
forecasts_lasso   <- read_parquet(file.path(here("output"), "forecasts.parquet")) %>% as.data.table()
  
#######################################################################################
############### R2_oos for ar3 model #############################

df <- forecasts_ar3 %>%
  mutate(date = as.Date(dplyr::lead(X_data$date, 1L))) %>%
  pivot_longer(
    cols = where(is.numeric),
    names_to = "target_stock",
    values_to = "forecast"
  ) %>%
  left_join(
    X_data %>%
      mutate(date = as.Date(date)) %>%
      pivot_longer(-date, names_to = "target_stock", values_to = "realized"),
    by = c("date", "target_stock")
  )

oos_r2 <- df %>%
  group_by(target_stock) %>%
  arrange(date_pred, .by_group = TRUE) %>%                 # ensure time order
  mutate(
    # expanding mean of realized up to t (exclude current date_pred):
    realized_lag = dplyr::lag(realized),
    csum   = cumsum(dplyr::coalesce(realized_lag, 0)),
    ccount = cumsum(!is.na(realized_lag)),
    naive_pred = ifelse(ccount > 0, csum / ccount, NA_real_)
  ) %>%
  # evaluate only where the benchmark is defined and forecast/realized exist
  filter(!is.na(naive_pred), !is.na(realized), !is.na(forecast)) %>%
  summarise(
    mse_model = mean((realized - forecast)^2),
    mse_naive = mean((realized - naive_pred)^2),
    r2_oos = 1 - mse_model / mse_naive,
    .groups = "drop"
  )

# --- Plot R2_oos ---

oos_r2_plot <- oos_r2 %>%
  mutate(sign = if_else(r2_oos >= 0, "Positive", "Negative"))

stats <- oos_r2_plot %>%
  summarise(
    mean_r2   = mean(r2_oos, na.rm = TRUE),
    n_pos     = sum(r2_oos >= 0, na.rm = TRUE),
    n_neg     = sum(r2_oos <  0, na.rm = TRUE),
    share_pos = mean(r2_oos >= 0, na.rm = TRUE),
    n_total   = sum(!is.na(r2_oos))
  )

p_benchmark <- ggplot(oos_r2_plot, aes(x = "", y = r2_oos)) +
  geom_jitter(
    width = 0.08, height = 0,
    color = "darkred", alpha = 0.6, size = 1.6, show.legend = FALSE
  ) +
  # dashed zero line
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  # mean line
  geom_hline(yintercept = stats$mean_r2, linewidth = 0.6, color = "steelblue") +
  # mean label
  annotate(
    "label",
    x = 1.03, y = stats$mean_r2,
    label = paste0("Mean: ", percent(stats$mean_r2, accuracy = 0.1)),
    hjust = 0, vjust = -0.4, size = 3, color = "steelblue"
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    x = NULL,
    y = "OOS R²",
    title = "Out-of-sample R² (vs. historical-mean benchmark)",
    subtitle = paste0(
      "N = ", scales::comma(stats$n_total),
      " | Positive = ", stats$n_pos,
      " | Negative = ", stats$n_neg,
      " | Share > 0 = ", percent(stats$share_pos, accuracy = 0.1)
    )
  ) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 20)
  )



#######################################################################################
############### R2_oos for lasso only #############################

X_long <- X_data %>%
  mutate(date = as.Date(date)) %>%
  tidyr::pivot_longer(-date, names_to = "target_stock", values_to = "realized")

df <- forecasts_lasso %>%
  mutate(date_pred = as.Date(date_pred)) %>%
  left_join(X_long, by = c("date_pred" = "date", "target_stock"))

oos_r2 <- df %>%
  group_by(target_stock) %>%
  arrange(date_pred, .by_group = TRUE) %>%                 # ensure time order
  mutate(
    # expanding mean of realized up to t (exclude current date_pred):
    realized_lag = dplyr::lag(realized),
    csum   = cumsum(dplyr::coalesce(realized_lag, 0)),
    ccount = cumsum(!is.na(realized_lag)),
    naive_pred = ifelse(ccount > 0, csum / ccount, NA_real_)
  ) %>%
  # evaluate only where the benchmark is defined and forecast/realized exist
  filter(!is.na(naive_pred), !is.na(realized), !is.na(forecast)) %>%
  summarise(
    mse_model = mean((realized - forecast)^2),
    mse_naive = mean((realized - naive_pred)^2),
    r2_oos = 1 - mse_model / mse_naive,
    .groups = "drop"
  )

# --- Plot R2_oos ---

oos_r2_plot <- oos_r2 %>%
  mutate(sign = if_else(r2_oos >= 0, "Positive", "Negative"))

stats <- oos_r2_plot %>%
  summarise(
    mean_r2   = mean(r2_oos, na.rm = TRUE),
    n_pos     = sum(r2_oos >= 0, na.rm = TRUE),
    n_neg     = sum(r2_oos <  0, na.rm = TRUE),
    share_pos = mean(r2_oos >= 0, na.rm = TRUE),
    n_total   = sum(!is.na(r2_oos))
  )

p_lasso <- ggplot(oos_r2_plot, aes(x = "", y = r2_oos)) +
  geom_jitter(
    width = 0.08, height = 0,
    color = "darkred", alpha = 0.6, size = 1.6, show.legend = FALSE
  ) +
  # dashed zero line
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  # mean line
  geom_hline(yintercept = stats$mean_r2, linewidth = 0.6, color = "steelblue") +
  # mean label
  annotate(
    "label",
    x = 1.03, y = stats$mean_r2,
    label = paste0("Mean: ", percent(stats$mean_r2, accuracy = 0.1)),
    hjust = 0, vjust = -0.4, size = 3, color = "steelblue"
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    x = NULL,
    y = "OOS R²",
    title = "Out-of-sample R² (vs. historical-mean benchmark)",
    subtitle = paste0(
      "N = ", scales::comma(stats$n_total),
      " | Positive = ", stats$n_pos,
      " | Negative = ", stats$n_neg,
      " | Share > 0 = ", percent(stats$share_pos, accuracy = 0.1)
    )
  ) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 20)
  )



# --- save results ---

ggsave(
  filename = "output/r2_oss.png",  # file path
  plot = p_benchmark,                                   # plot object
  width = 8, height = 5, dpi = 300            # dimensions & resolution
)

ggsave(
  filename = "output/r2_oss.png",  # file path
  plot = p_lasso,                                   # plot object
  width = 8, height = 5, dpi = 300            # dimensions & resolution
)