% Distributed under GPL 2 or later

\name{distmc}
\alias{distmc}
\title{Distance Matrix Computation with multi-threads}
\description{
  This function is similar to \code{dist()} in \pkg{stats}, with additional support of multi-threading.
}
\usage{
distmc(x, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)
}
\arguments{
  \item{x}{a numeric matrix, data frame or \code{"dist"} object.}
  \item{method}{the distance measure to be used. This must be one of
    \code{"euclidean"}, \code{"maximum"}, \code{"manhattan"},
    \code{"canberra"}, \code{"binary"} or \code{"minkowski"}.
    Any unambiguous substring can be given.}
  \item{diag}{logical value indicating whether the diagonal of the
    distance matrix should be printed by \code{print.dist}.}
  \item{upper}{logical value indicating whether the upper triangle of the
    distance matrix should be printed by \code{print.dist}.}
  \item{p}{The power of the Minkowski distance.}
}
\details{
  Available distance measures are (written for two vectors \eqn{x} and
  \eqn{y}):
  \describe{
    \item{\code{euclidean}:}{Usual square distance between the two
      vectors (2 norm).}

    \item{\code{maximum}:}{Maximum distance between two components of \eqn{x}
      and \eqn{y} (supremum norm)}

    \item{\code{manhattan}:}{Absolute distance between the two vectors
      (1 norm).}

    \item{\code{canberra}:}{
      \eqn{\sum_i |x_i - y_i| / |x_i + y_i|}{sum(|x_i - y_i| / |x_i + y_i|)}.
      Terms with zero numerator and denominator are omitted from the sum
      and treated as if the values were missing.

      This is intended for non-negative values (e.g. counts): taking the
      absolute value of the denominator is a 1998 \R modification to
      avoid negative distances.
    }

    \item{\code{binary}:}{(aka \emph{asymmetric binary}): The vectors
      are regarded as binary bits, so non-zero elements are \sQuote{on}
      and zero elements are \sQuote{off}.  The distance is the
      \emph{proportion} of bits in which only one is on amongst those in
      which at least one is on.}

    \item{\code{minkowski}:}{The \eqn{p} norm, the \eqn{p}th root of the
      sum of the \eqn{p}th powers of the differences of the components.}
  }

  Missing values are allowed, and are excluded from all computations
  involving the rows within which they occur.
  Further, when \code{Inf} values are involved, all pairs of values are
  excluded when their contribution to the distance gave \code{NaN} or
  \code{NA}.\cr
  If some columns are excluded in calculating a Euclidean, Manhattan,
  Canberra or Minkowski distance, the sum is scaled up proportionally
  to the number of columns used.  If all pairs are excluded when calculating a
  particular distance, the value is \code{NA}.

  The \code{"distmc"} method of \code{as.matrix()} and \code{as.dist()}
  can be used for conversion between objects of class \code{"dist"}
  and conventional distance matrices.

  \code{as.dist()} is a generic function.  Its default method handles
  objects inheriting from class \code{"dist"}, or coercible to matrices
  using \code{as.matrix()}.  Support for classes representing
  distances (also known as dissimilarities) can be added by providing an
  \code{as.matrix()} or, more directly, an \code{as.dist} method
  for such a class.
}
\value{
  \code{distmc} returns an object of class \code{"dist"}.

  The lower triangle of the distance matrix stored by columns in a
  vector, say \code{do}. If \code{n} is the number of
  observations, i.e., \code{n <- attr(do, "Size")}, then
  for \eqn{i < j \le n}, the dissimilarity between (row) i and j is
  \code{do[n*(i-1) - i*(i-1)/2 + j-i]}.
  The length of the vector is \eqn{n*(n-1)/2}, i.e., of order \eqn{n^2}.

  The object has the following attributes (besides \code{"class"} equal
  to \code{"dist"}):
  \item{Size}{integer, the number of observations in the dataset.}
  \item{Labels}{optionally, contains the labels, if any, of the
    observations of the dataset.}
  \item{Diag, Upper}{logic, corresponding to the arguments \code{diag}
    and \code{upper} above, specifying how the object should be printed.}
  \item{call}{optional, the \code{call} used to create the
    object.}
  \item{method}{optional, the distance measure used; resulting from
    \code{\link{distmc}()}, the (\code{match.arg()}ed) \code{method}
    argument.}
}
\references{
  Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988)
  \emph{The New S Language}.
  Wadsworth & Brooks/Cole.

  Mardia, K. V., Kent, J. T. and Bibby, J. M. (1979)
  \emph{Multivariate Analysis.} Academic Press.

  Borg, I. and Groenen, P. (1997)
  \emph{Modern Multidimensional Scaling.  Theory and Applications.}
  Springer.
}
\seealso{
  \code{dist()} in the \pkg{stats} package
}
\examples{
data(iris)
df<-data.frame(iris[-5])
dist.data<-distmc(df, 'manhattan')
}
\keyword{distmc}
\keyword{lof}
