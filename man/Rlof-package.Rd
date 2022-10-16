\name{Rlof-package}
\alias{Rlof-package}
\alias{Rlof}
\docType{package}
\title{
R Parallel Implementation of Local Outlier Factor(LOF)
}
\description{
R parallel implementation of Local Outlier Factor(LOF) which uses multiple CPUs to significantly speed up the LOF computation for large datasets. (Note: The overall performance depends on the computers especially the number of the cores).It also supports multiple k values to be calculated in parallel, as well as various distance measures in addition to the default Euclidean distance.
}
\details{
\tabular{ll}{
Package: \tab Rlof\cr
Version: \tab 1.1.0\cr
Date: \tab 2015-09-10\cr
Depends: \tab R (>= 2.14.0), doParallel, foreach\cr
License: \tab GPL (>= 2)\cr
URL: \tab \url{https://CRAN.R-project.org/package=Rlof}\cr
What's new: \tab bug fix
}
}


\author{
Yingsong Hu <yingsonghu@hotmail.com>, Wayne Murray and Yin Shan, Australia.

Maintainer: Yingsong Hu <yingsonghu@hotmail.com>
}
\keyword{ package }
