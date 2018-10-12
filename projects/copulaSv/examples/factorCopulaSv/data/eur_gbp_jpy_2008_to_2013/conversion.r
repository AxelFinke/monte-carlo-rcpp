setwd("/home/axel/Dropbox/research/code/cpp/mc/copulaSv/examples/factorCopulaSv/data/eur_gbp_jpy_2008_to_2013/")
dataAux <- read.csv("RealDataExchangeRates.csv")

K <- 3
logExchangeRates <- t(dataAux[,1:K]) # log-exchange rates
logVolatilities  <- t(dataAux[,(K+1):(2*K)]) # log-exchange rates

write.table(logExchangeRates, "logExchangeRates.dat")
write.table(logVolatilities, "logVolatilities.dat")
write.table(logExchangeRates[,1], "initialLogExchangeRates.dat")
write.table(logVolatilities[,1], "initialLogVolatilities.dat")
