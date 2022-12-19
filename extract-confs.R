load("log.Rdata")
rs <- iraceResults
nIterations <- length(rs$allElites)
confs <- rs$allElites[[nIterations]]
confs <- c(1, confs)
elites <- rs$allConfigurations[confs,]
write.csv(elites, "confs.csv", row.names=FALSE)
