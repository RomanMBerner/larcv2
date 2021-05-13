/**
 * \file StackTensor3D.h
 *
 * \ingroup ImageMod
 *
 * \brief Class def header for a class StackTensor3D
 *
 * @author francois
 */

/** \addtogroup ImageMod

    @{*/
#ifndef __StackTensor3D_H__
#define __StackTensor3D_H__

#include "larcv/core/Processor/ProcessBase.h"
#include "larcv/core/Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class StackTensor3D ... these comments are used to generate
     doxygen documentation!
  */
  class StackTensor3D : public ProcessBase {

  public:

    /// Default constructor
    StackTensor3D(const std::string name="StackTensor3D");

    /// Default destructor
    ~StackTensor3D(){}

    void configure_labels(const PSet&);

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    enum PoolType_t {kMinPool,kMaxPool,kSumPool};

  private:
    std::vector<std::string> _output_producer_v;
    std::vector<std::string> _tensor3d_producer_v;
    std::string _reference_producer;
    std::vector<unsigned short> _pool_type_v;
    double _fuzzy_distance;
    size_t _num_tensor;
  };

  /**
     \class larcv::StackTensor3DFactory
     \brief A concrete factory class for larcv::StackTensor3D
  */
  class StackTensor3DProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    StackTensor3DProcessFactory() { ProcessFactory::get().add_factory("StackTensor3D",this); }
    /// dtor
    ~StackTensor3DProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new StackTensor3D(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group
