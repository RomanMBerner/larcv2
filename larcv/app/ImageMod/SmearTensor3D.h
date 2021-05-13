/**
 * \file SmearTensor3D.h
 *
 * \ingroup ImageMod
 *
 * \brief Class def header for a class SmearTensor3D
 *
 * @author francois
 */

/** \addtogroup ImageMod

    @{*/
#ifndef __SmearTensor3D_H__
#define __SmearTensor3D_H__

#include "larcv/core/Processor/ProcessBase.h"
#include "larcv/core/Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class SmearTensor3D ... these comments are used to generate
     doxygen documentation!
  */
  class SmearTensor3D : public ProcessBase {

  public:

    /// Default constructor
    SmearTensor3D(const std::string name="SmearTensor3D");

    /// Default destructor
    ~SmearTensor3D(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  private:

    void configure_labels(const PSet& cfg);

    std::vector<std::string> _target_producer_v;
    std::vector<std::string> _output_producer_v;

    std::vector<float> _voxel_smear_v;
  };

  /**
     \class larcv::SmearTensor3DFactory
     \brief A concrete factory class for larcv::SmearTensor3D
  */
  class SmearTensor3DProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    SmearTensor3DProcessFactory() { ProcessFactory::get().add_factory("SmearTensor3D",this); }
    /// dtor
    ~SmearTensor3DProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new SmearTensor3D(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group
