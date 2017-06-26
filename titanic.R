library(dplyr)
library(ggplot2)
library(ggthemes)
library(scales)
library(randomForest)


trn_data <- read.csv("train.csv")
tst_data <- read.csv("test.csv")
full <- bind_rows(trn_data, tst_data)

###################################################################################################
# Feature engineering
###################################################################################################

#Creating title column
  full$Title<- full$Name
  full$Title <- gsub("(\\..*)", "", x=full$Title)
  full$Title <- gsub("(.*,)"  , "", x=full$Title)
  full$Title <- gsub(" ","",x=full$Title)
#alternatively
full$Title <-gsub(" ","", sapply(full$Name, function(x) strsplit(x, split='[,.]')[[1]][2]) )
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])

# sapply( # function from apply family, simplifying 
#   full$Name, # field to which the below function will be applied
#   function(x) strsplit(x, split='[,.]')[[1]][2]) # function that will be applied to the field, splits x by , and . and returns a length 1 list
#                                                  # [[1]] for the first entry in a list, [1] is surname, [2] title

# checking Title groups
full %>%
  group_by(Title) %>%
  summarize(n=n(), m = mean(Age)) %>%
  arrange(desc(n))

#Normalizing titles
full$Title[full$Title == "Mlle"] <- "Miss"
full$Title[full$Title == "Ms"] <- "Miss"
full$Title[full$Title == "Mme"] <- "Mrs"
rare_titles <- c('Dona', 'Lady', 'theCountess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
full$Title[full$Title %in% rare_titles] <- 'Rare Title'

#overview of dataset
str(full)

#determining family size
full$FamilySize <- full$SibSp + full$Parch +1 # Family size as a sum of Siblings, Spouses, Parents and children + the person for whom the record is
ggplot(data=full[full$Survived %in% c(0,1),], aes(x=FamilySize, fill = factor(Survived))) + 
  geom_bar(stat = "count", position="dodge") + 
  scale_x_continuous(breaks=c(1:11)) +
  labs(X="Family size") +
  theme_few()

#Grouping family sizes and visuazlizing as bar and as mosaic plot
full$FSizeGroup <- ifelse(full$FamilySize==1,"Singleton", ifelse(full$FamilySize<5, "Small","Large"))
ggplot(data=full[full$Survived %in% c(0,1),], aes(x=FSizeGroup, fill = factor(Survived))) + 
  geom_bar(stat = "count", position="dodge") + 
  labs(X="Family size") +
  theme_few()
mosaicplot(table(full$FSizeGroup, full$Survived),main="Family group by survival",shade=T)
# mosaic plot shows that small families have the highest chance of survival and 
# family size > 4 negatively affects chances of survival

#Determining deck (distance from lifeboats)
unique(full$Cabin)
full$Deck <- sapply(full$Cabin, function(x) strsplit(x,NULL)[[1]][1])

###################################################################################################
#Imputation / Guessing / Predicting missing values
###################################################################################################

#Embarkment - 2 missing - id 62 & 830
full %>%
  group_by(Embarked) %>%
  summarize(n=n())
fullTemp <- full %>% filter(Embarked != "") # getting rid of missing values to estimate them based on fare and passenger class
ggplot(data=fullTemp, aes(x=Embarked, y=Fare, fill = factor(Pclass))) +
  geom_boxplot() + 
  geom_hline(aes(yintercept=80), colour='red', linetype='dashed', lwd=1) + # 80 because missing passengers paid 80 for ticket
  scale_y_continuous(labels=dollar_format()) + 
  theme_few()
# median ticket price in C (Charbourg) is 80, assuming that's where the 2 passengers got on
full$Embarked[c(62,830)] <- "C"

# Ticket price - 1 missing - id 1044, 17 paid nothing
summary(full$Fare)
full[1044,] #Embarked in S, class 3
fullTemp <- full %>% filter(is.na(Fare)==F & Embarked=="S" & Pclass==3)
fullTemp %>%
  summarize(avg=mean(Fare), med = median(Fare)) #mean fare 14.43, median 8.05
ggplot(data=fullTemp, aes(x=Fare)) + 
  geom_density(fill="blue", alpha=0.5) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),colour='red', linetype='dashed', lwd=1) + 
  scale_x_continuous(labels=dollar_format()) + 
  theme_few()
full$Fare[1044] <- median(fullTemp$Fare)

#0 fare
aggregate(Fare~Pclass, data=full, FUN=median)
full$Fare[full$Fare==0 & full$Pclass==1] <- median(full$Fare[full$Pclass==1])
full$Fare[full$Fare==0 & full$Pclass==2] <- median(full$Fare[full$Pclass==2])
full$Fare[full$Fare==0 & full$Pclass==3] <- median(full$Fare[full$Pclass==3])

# multiple linear regression for age - start with factoring Title and Sex
full$TitleF <- factor(full$Title, levels = c('Mr', 'Miss', 'Mrs', 'Master', 'Rare Title'), labels = c(1,2,3,4,5))
full$SexF <- factor(full$Title, levels = c('male','female'), labels = c(1,2))
fullTemp <- full[is.na(full$Age)==F,] # data frame without empty ages
emp <- full[is.na(full$Age)==T,] # data frame with empty ages

reg <- lm(data=full, formula = Age ~ Fare + Title + SibSp + Sex)
summary(reg) # repeat the two lines until model is omptimized
pred <- predict(reg, newdata=emp)
summary(pred)
#there is one negative value, flipping to positive
full$Age[is.na(full$Age)==T] <- pred # assign predicted values to original data frame
full$Age <- ifelse(full$Age<0, full$Age*-1, full$Age)
# Histogram for Age because why not
ggplot(data=full, aes(x=Age)) + 
  geom_histogram(binwidth=2, fill ='orange', colour='black') + 
  theme_solarized_2()

###################################################################################################
# Feature engineering 2 now that we have preprocessing done and data isn't missing
###################################################################################################
full$Child <- ifelse(full$Age<18,'Child','Adult')
full$Mother <- ifelse(full$Sex == 'female' & full$Age>=18 & full$Parch>0 & full$Title != 'Miss','Mother','Not mother')
table(full$Mother, full$Survived)
full$Child <- factor(full$Child)
full$Mother <- factor(full$Mother)

###################################################################################################
# Model building
###################################################################################################
full$Embarked <- factor(full$Embarked)
full$Title <- factor(full$Title)
full$FSizeGroup <- factor(full$FSizeGroup)
full$Pclass <- factor(full$Pclass)
full$Survived <- factor(full$Survived)
train <- full[is.na(full$Survived)==F,]
test <- full[is.na(full$Survived)==T,]
str(train)
rf <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FSizeGroup + Child + Mother, data=train, ntree=100)
# rf <- randomForest(Survived ~ Pclass + Sex + Age + Fare + Title , data=train, ntree=100) #possibility for later change

#error rate
plot(rf, ylim = c(0,0.36))
legend('topright', colnames(rf$err.rate), col=1:3, fill=1:3)

#get importance (p-value equivalent)
importance <- importance(rf)
varImportance <- data.frame(Variables = row.names(importance), Importance = round(importance[,'MeanDecreaseGini'],2))

#ranking performance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

#visualize with ggplot
ggplot(rankImportance, aes(x=reorder(Variables, Importance), y=Importance, fill = Importance)) + 
  geom_bar(stat='identity') + 
  geom_text(aes(x=Variables, y=0.5, label=Rank), hjust=0, vjust=0.55, size=4, colour='red') +
  labs(x='Variables') + 
  coord_flip() + 
  theme_few()

###################################################################################################
# Prediction
###################################################################################################
tst_pred <- predict(rf, newdata=test, type='response')
solution <- data.frame(PassengerID = test$PassengerId, Survived = tst_pred)
write.csv(solution, file = 'res_final.csv', row.names = F)