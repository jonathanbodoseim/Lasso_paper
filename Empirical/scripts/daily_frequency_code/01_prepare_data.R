# Load packages
library(vroom)
library(here)
library(stringr)     
library(stringi)


##################################################################################
# Load the CSV file using a project-relative path
df <- vroom(here("data", "all-the-news-2-1.csv"))

# Exclude all publications with infrequent output
df <- df %>% 
  filter(publication %in% c("The New York Times", "Reuters", "CNBC"))

# Remove empty rows
df <- df[!is.na(df$article), ]

# Count the number of words in each article
df$word_count <- str_count(df$article, "\\S+")

# Filter out articles with fewer than 100 words
df <- df[df$word_count >= 100, ]
df <- df[df$word_count <= 4000, ]


# Remove all articles with irrelevant section or publication after checking all those with more than 100 entries
sections_to_remove <- c("arts", "sports", "books", "fashion", "Restaurants", "dining", 
                        "Drug and Substance Abuse", "Drugs", "Drama", "Divorce", 
                        "style", "Lifestyle", "tv", "theater", "entertainment", 
                        "Noisey", "Entertainment News", "Food by VICE", "Music by VICE", 
                        "movies", "nyregion", "Identity", "obituaries", "learning", "travel",
                        "opinion", "t-magazine", "magazine", "crosswords", "science", "Sports News",
                        "CNBC News Releases", "smarter-living", "obituary", "reader-center", "books-and-arts",
                        "opinions", "Personal Finance", "Invest in You: Ready. Set. Grow.", "Trading Nation",
                        "your-money", "Investor Toolkit", "Straight Talk", "Retirment", "Advisor Insight")


df <- df[!df$section %in% sections_to_remove, ]

# Check for duplicates and remove them
duplicates <- df[duplicated(df$article), ]
df <- df[!duplicated(df$article), ]
rm(duplicates)

# Define list of stock market relevant terms and filter for articles with at least one mention
terms <- "stock market|financial market|bond market|stock exchange|equity market|securities market|bull market|bear market|market rally|market crash|
          stock market crash|market correction|market surge|market slump|market downturn|market uptrend|
          market rebound|market recovery|market volatility|derivatives market|bond yield|stock return"

df <- df[stri_detect_regex(df$article, terms, case_insensitive = TRUE), ]

# Remove Jim Cramer investor columns
df <- df %>%
  filter(!grepl("\\bCramer\\b", title, ignore.case = TRUE))

df <- df %>%
  mutate(date = as.Date(date)) %>%  # Extract only the date part
  arrange(date) %>%  # Ensure data is sorted by date
  mutate(day = dense_rank(date))   # Assign a unique number to each unique date

# Add an article id 
df$article_id <- seq(1, nrow(df))

##################################################################################
# Save data
save(df, file = here("data", "clean_articles.rds"))
##################################################################################




