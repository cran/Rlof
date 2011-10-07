% Distributed under GPL 2 or later
\name{lof}
\alias{lof}
\title{ Local Outlier Factor }
\description{ A function that finds the local outlier factor (Breunig et al.,2000) of
the matrix "data" using k neighbours. The local outlier factor (LOF) is a measure of outlierness
that is calculated for each observation. The user decides whether or not an observation 
will be considered an outlier based on this measure. The LOF takes into consideration
the density of the neighbourhood around the observation to determine its outlierness. This 
is a faster implementation of LOF by using a different data structure and distance calculation function compared to \code{lofactor()} function available in \pkg{dprep} package. It also supports multiple k values to be calculated in parallel, as well as various distance measures besides the default Euclidean distance. 
  }
\usage{
lof(data, k, ...)
}
\arguments{
  \item{data}{ The data set to be explored, which can be a data.frame or matrix}
  \item{k}{ The kth-distance to be used to calculate LOFs. k can be a vector which contains multiple k values based on which LOFs need to be calculated.}
  \item{...}{ The parameters to be passed to \code{\link{distmc}()} function, specifying the distance measure.}
}
\details{
  The LOFs are calculated over multiple k values in parallel, and the maximum number of the cpus will be utilised to achieve the best performance.
}
\value{
  \item{lof}{ A matrix with the local outlier factor of each observation as rows and each k value as columns}
}
\references{ Breuning, M., Kriegel, H., Ng, R.T, and Sander. J. (2000). 
LOF: Identifying density-based local outliers. In Proceedings of the ACM SIGMOD 
International Conference on Management of Data.}
\author{Yingsong Hu, Wayne Murray and Yin Shan with Strategic Data Mining Team, Department of Human Services, Australian.}
\examples{
#---- Detecting the top outliers using the LOF algorithm 
#---- with k = 5,6,7,8,9 and 10, respectively----
data(iris)
df<-data.frame(iris[-5])
df.lof<-lof(df,c(5:10))
}
\keyword{Rlof}
\keyword{lof}