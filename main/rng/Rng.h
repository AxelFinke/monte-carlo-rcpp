/// \file
/// \brief A class for random number generation.
///
/// This file contains the Rng class which generates random numbers from 
/// various distributions.

#ifndef __RNG_H
#define __RNG_H

#include <RcppArmadillo.h>
#include <vector>
#include <random>

/// A base class used to permit pointers an instantiation of RngDerived
/// and its members.
class Rng
{
  
public:
  
  virtual void setSeed(const unsigned long int seed) = 0;
  virtual void randomiseSeed() = 0;
  virtual unsigned long int & getSeed() = 0;
  virtual bool randomBernoulli(const double prob) = 0;
  virtual unsigned long int randomBinomial(const double range, const double prob) = 0;
  virtual unsigned long int randomDiscrete(std::vector<double>& weights) = 0;
  virtual double randomExponential(const double rate) = 0;
  virtual double randomFisher(const double df1, const double df2) = 0;
  virtual double randomGeometric(const double prob) = 0;
  virtual double randomGamma(const double shape, const double scale) = 0;
  virtual double randomLognormal(const double location, const double scale) = 0;
  virtual double randomNormal(const double mean, const double stdDev) = 0;
  virtual unsigned long int randomPoisson(const double mean) = 0;
  virtual double randomStudent(const double df) = 0;
  virtual int randomUniformInt(const int from, const int thru) = 0;
  virtual double randomUniformReal(const double from, const double to) = 0;
  //virtual ~Rng() {} // TODO
  
};

/// \brief A class template for dealing with random number generation.
/// \param T an engine, e.g. T is std::mt19937 or some other engine type.
template <typename T> class RngDerived : public Rng
{
  
public:
  
  //////////////////////////////////////////////////////////////////////////////
  // Constructors and Destructor
  //////////////////////////////////////////////////////////////////////////////
  
  /// Initialises the random number generator with engine specified by the user
  /// and with a pseudo-random system-generated seed.
  RngDerived(T& engine);
  /// Initialises the random number generator with an engine and seed
  /// specified by the user.
  RngDerived(T& engine, const unsigned long int seed);
  // Destructor
  //~RngDerived(); // TODO
  
  //////////////////////////////////////////////////////////////////////////////
  // Member functions for setting or obtaining information about engine/seed
  //////////////////////////////////////////////////////////////////////////////
  
  /// Changes the seed.
  void setSeed(const unsigned long int seed);
  /// Randomises the seed.
  void randomiseSeed();
  /// Returns the engine.
  T& getEngine();
  /// Returns the seed.
  unsigned long int & getSeed();
  
  //////////////////////////////////////////////////////////////////////////////
  // Member functions for sampling from specific parametrised distributions
  //////////////////////////////////////////////////////////////////////////////
  
  /// Returns a random number from a Bernoulli distribution with specified
  /// success probability.
  bool randomBernoulli(const double prob);
  /// Returns a random number from a binomial distribution with specified
  /// range and success probability parameters.
  unsigned long int randomBinomial(const double range, const double prob);
  /// Returns a random number from a discrete distribution on 
  /// \f$\{0, 1, 2, ..., \mathit{weights.size()}-1\}\f$ for a specified weight vector.
  unsigned long int randomDiscrete(std::vector<double>& weights);
  /// Returns a random number from an exponential distribution with specified
  /// rate parameter.
  double randomExponential(const double rate);
  /// Returns a random number from Fisher's F-distribution with specified
  /// degrees-of-freedom parameters.
  double randomFisher(const double df1, const double df2);
  /// Returns a random number from a geometric distribution on 
  /// \f$\{0, 1, 2, \dotsc \}\f$ with specified success probability.
  double randomGeometric(const double prob);
  /// Returns a random number from a gamma distribution with specified shape
  /// and scale parameters.
  double randomGamma(const double shape, const double scale);
  /// Returns a random number from a lognormal distribution with specified 
  /// location and scale parameters.
  double randomLognormal(const double location, const double scale);
  /// Returns a random number from a normal distribution with specified mean 
  /// and standard deviation.
  double randomNormal(const double mean, const double stdDev);
  /// Returns a random number from a Poisson distribution with specified mean.
  unsigned long int randomPoisson(const double mean);
  /// Returns a random number from a Student-t distribution with specified 
  /// degrees-of-freedom parameter.
  double randomStudent(const double df);
  /// Returns a random number from a Uniform distribution on 
  /// \f$\{\mathit{from}, \mathit{from}+1, \dotsc, \mathit{thru}-1, \mathit{thru}\}\f$.
  int randomUniformInt(const int from, const int thru);
  /// Returns a random number from a Uniform distribution on 
  /// \f$[\mathit{from}, \mathit{to}]\f$.
  double randomUniformReal(const double from, const double to);
  
private:

  unsigned long int seed_;
  T& engine_;
  
};

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

/// Initialises the random number generator with engine specified by the user
/// and with a pseudo-random system-generated seed.
template <class T> 
RngDerived<T>::RngDerived(T& engine) 
: seed_(std::random_device{}()), engine_(engine) 
{
  engine_.seed(seed_); 
}
/// Initialises the random number generator with an engine and seed
/// specified by the user.
template <class T> 
RngDerived<T>::RngDerived(T& engine, const unsigned long int seed) 
: seed_(seed), engine_(engine)
{ 
  engine_.seed(seed_); 
} 
// Destructor.
//~RngDerived(); // TODO
  
  
////////////////////////////////////////////////////////////////////////////////
// Member functions for setting or obtaining information about engine/seed
////////////////////////////////////////////////////////////////////////////////

/// Changes the seed.
template <class T> 
void RngDerived<T>::setSeed( const unsigned long int seed ) 
{ 
  seed_ = seed;
  engine_.seed(seed_); 
}
/// Randomises the seed.
template <class T> 
void RngDerived<T>::randomiseSeed( ) 
{
  seed_ = std::random_device{}(); 
  engine_.seed(seed_); 
} 
/// Returns the engine.
template <class T> 
T& RngDerived<T>::getEngine()
{ 
  return engine_; 
} 
/// Returns the seed.
template <class T> 
unsigned long int & RngDerived<T>::getSeed() 
{ 
  return seed_;   
}


////////////////////////////////////////////////////////////////////////////////
// Member functions for sampling from specific parametrised distributions
////////////////////////////////////////////////////////////////////////////////

/// Returns a random number from a Bernoulli distribution with specified
/// success probability.
template <class T> 
bool RngDerived<T>::randomBernoulli(const double prob)
{
  static std::bernoulli_distribution d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(prob) );
}
/// Returns a random number from a binomial distribution with specified
/// range and success probability parameters.
template <class T> 
unsigned long int RngDerived<T>::randomBinomial(const double range, const double prob)
{
  static std::binomial_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(range, prob) );
}
/// Returns a random number from a discrete distribution on 
/// \f$\{0, 1, 2, ..., \mathit{weights.size()}-1\}\f$ for a specified weight vector.
template <class T> 
unsigned long int RngDerived<T>::randomDiscrete(std::vector<double>& weights)
{
  //static std::discrete_distribution<> d{};
  //using parameterType = decltype(d)::param_type;
  //return d( getEngine(), parameterType(weights) );
  static std::discrete_distribution<unsigned long int> d(weights.begin(), weights.end());
  return d(getEngine());
}
/// Returns a random number from an exponential distribution with specified
/// rate parameter.
template <class T> 
double RngDerived<T>::randomExponential(const double rate)
{
  static std::exponential_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(rate) );
}
/// Returns a random number from Fisher's F-distribution with specified
/// degrees-of-freedom parameters.
template <class T> 
double RngDerived<T>::randomFisher(const double df1, const double df2)
{
  static std::fisher_f_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(df1, df2) );
}
/// Returns a random number from a geometric distribution on 
/// \f$\{0, 1, 2, \dotsc \}\f$ with specified success probability.
template <class T> 
double RngDerived<T>::randomGeometric(const double prob)
{
  static std::geometric_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(prob) );
}
/// Returns a random number from a gamma distribution with specified shape
/// and scale parameters.
template <class T> 
double RngDerived<T>::randomGamma(const double shape, const double scale)
{
  static std::gamma_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(shape, scale) );
}
/// Returns a random number from a lognormal distribution with specified 
/// location and scale parameters.
template <class T> 
double RngDerived<T>::randomLognormal(const double location, const double scale)
{
  static std::lognormal_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(location, scale) );
}
/// Returns a random number from a normal distribution with specified mean 
/// and standard deviation.
template <class T>
double RngDerived<T>::randomNormal(const double mean, const double stdDev)
{
  static std::normal_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(mean, stdDev) );
}
/// Returns a random number from a Poisson distribution with specified mean.
template <class T> 
unsigned long int RngDerived<T>::randomPoisson(const double mean)
{
  static std::poisson_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(mean) );
}
/// Returns a random number from a Student-t distribution with specified 
/// degrees-of-freedom parameter.
template <class T> 
double RngDerived<T>::randomStudent(const double df)
{
  static std::student_t_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(df) );
}
/// Returns a random number from a Uniform distribution on 
/// \f$\{\mathit{from}, \mathit{from}+1, \dotsc, \mathit{thru}-1, \mathit{thru}\}\f$.
template <class T> 
int RngDerived<T>::randomUniformInt(const int from, const int thru)
{
  static std::uniform_int_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(from, thru) );
}
/// Returns a random number from a Uniform distribution on 
/// \f$[\mathit{from}, \mathit{to}]\f$.
template <class T> 
double RngDerived<T>::randomUniformReal(const double from, const double to)
{
  static std::uniform_real_distribution<> d{};
  using parameterType = decltype(d)::param_type;
  return d( getEngine(), parameterType(from, to) );
}

#endif