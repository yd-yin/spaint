/**
 * rafl: QuantitativePerformance.h
 */

#ifndef H_RAFL_QUANTITATIVEPERFORMANCE
#define H_RAFL_QUANTITATIVEPERFORMANCE

namespace rafl {

/**
 * \brief An instance of this class represents the quantitative performance obtained by an algorithm.
 */
class QuantitativePerformance
{
  //#################### NESTED TYPES ####################
private:
  /**
   * \brief An instance of this struct represents a performance measure.
   */
  struct Measure
  {
    //~~~~~~~~~~~~~~~~~~~~ PUBLIC VAIRABLES ~~~~~~~~~~~~~~~~~~~~

    /** The mean of the measure. */
    float mean;

    /** The standard deviation of the measure. */
    float stdDev;
  };

  /** The accuracy measure. */
  Measure m_accuracy;

  /** The number of samples used to generate the performance measures. */
  size_t m_samples;

  //#################### PUBLIC CONSTRUCTOR ####################
public:
  /**
   * \brief Constructs a quantitative performance measure from a single sample.
   *
   * \param accuracy   The accuracy measured from one sample.
   */
  explicit QuantitativePerformance(float accuracy)
  : m_samples(1)
  {
    m_accuracy.mean = accuracy;
    m_accuracy.stdDev = 0.0f;
  }

  //#################### PRIVATE CONSTRUCTOR ####################
private:
  /**
   * \brief Constructs an arbitrary quantitative performance measure.
   *
   * \param accuracy   The accuracy measured from one sample.
   * \param stdDev     The standard deviation of the accuracy samples.
   * \param samples    The number of samples used to calculate the standard deviation. 
   */
  QuantitativePerformance(float accuracy, float stdDev, float sampleCount)
  : m_samples(sampleCount)
  {
    m_accuracy.mean = accuracy;
    m_accuracy.stdDev = stdDev;
  }

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Constructs a single quantitative perofrmance measure form a vector of them.
   *
   * \param qpv    A vector of quantitative performance measures.
   */
  static QuantitativePerformance average(const std::vector<QuantitativePerformance> qpv)
  {
    size_t sampleCount = qpv.size();

    float sumMean = 0;
    for(size_t i = 0; i < sampleCount; ++i)
    {
      sumMean += qpv[i].mean_accuracy();
    }
    float mean = sumMean / sampleCount;

    float sumVariance = 0;
    for(size_t i = 0; i < sampleCount; ++i)
    {
      sumVariance += pow(mean - qpv[i].mean_accuracy(), 2);
    }
    float variance = sqrt(sumVariance / sampleCount);

    return QuantitativePerformance(mean, variance, sampleCount);
  }

  /**
   * \brief Getter function for mean accuracy.
   *
   * \return The mean accuracy measure.
   */
  float mean_accuracy() const
  {
    return m_accuracy.mean;
  }

  /**
   * \brief Prints a tab delineated header to a stream.
   *
   * \param out  The stream.
   * \return     The stream.
   */
  std::ostream& print_accuracy_header(std::ostream& out) const
  {
    out << "Acc" << "\t" << "Std";
    return out;
  }

  /**
   * \brief Prints tab delineated accuracy values to a stream.
   *
   * \param out  The stream.
   * \return     The stream.
   */
  std::ostream& print_accuracy_values(std::ostream& out) const
  {
    out << m_accuracy.mean << "\t" << m_accuracy.stdDev;
    return out;
  }

  /**
   * \brief Binds a quantitative performance object to a stream.
   *
   * \param out   The stream.
   * \param qp    The quantitative performance object.
   * \return      The stream.
   */
  friend std::ostream& operator<<(std::ostream& out, const QuantitativePerformance& qp)
  {
    out << "accuracy: " << qp.m_accuracy.mean << " +/- " << qp.m_accuracy.stdDev << ", samples: " << qp.m_samples << "\n";
    return out;
  }
};

} //end namespace rafl

#endif

