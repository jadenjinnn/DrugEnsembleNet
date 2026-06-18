library(pROC)
library(yardstick)

predictions_df <- read.csv("./data/results/borda/MASTER_DRUG_RANKINGS.csv")

true_positives <- c("DB17924", "DB00470", "DB00148", "DB21549", "DB16975", "DB04844", "DB11947", "DB06819", "DB11725", "DB01017", "DB01043", "DB14509", "DB00313", "DB06685", "DB08387", "DB09270", "DB13978", "DB15155", "DB01156", "DB00915", "DB12161", "DB11340", "DB02709", "DB18165", "DB17870", "DB00334", "DB13025", "DB21645", "DB01065", "DB05565", "DB16977", "DB01039", "DB12116", "DB11677", "DB00740", "DB16968", "DB08887", "DB00734", "DB00514", "DB00908", "DB00289", "DB00215", "DB11915")

lengh(true_positives)

predictions_df$label <- ifelse(predictions_df$DrugBank_ID %in% true_positives, 1, 0)
predictions_df$label_factor <- factor(predictions_df$label, levels = c("1", "0"))

roc_obj <- roc(predictions_df$label, predictions_df$Borda_Score, quiet = TRUE)
print(paste("ROC-AUC:", round(auc(roc_obj), 4)))

pr_result <- pr_auc(predictions_df, truth = label_factor, Borda_Score)
print("Precision-Recall AUC:")
print(pr_result)

k <- 100

sorted_df <- predictions_df[order(-predictions_df$Borda_Score), ]
top_k_df <- head(sorted_df, k)

observed_positives <- sum(top_k_df$label == 1)
total_positives <- sum(predictions_df$label == 1)
total_drugs <- nrow(predictions_df)

expected_positives <- k * (total_positives / total_drugs)
fold_enrichment <- observed_positives / expected_positives

message("=============================")
message("      TOP 100 ANALYSIS       ")
message("=============================")
message("Total Drugs in Database:   ", total_drugs)
message("Actual hits in Top 100:    ", observed_positives)
message("Expected by chance:        ", round(expected_positives, 2))
message("Fold Enrichment:           ", round(fold_enrichment, 2), "x")
message("=============================")

# Calculate Hypergeometric p-value [cite: 87-93]
p_value <- phyper(observed_positives - 1, 
                  total_positives, 
                  total_drugs - total_positives, 
                  k, 
                  lower.tail = FALSE)

message("Hypergeometric p-value: ", signif(p_value, 4))