#ifndef __StackTensor3D_CXX__
#define __StackTensor3D_CXX__

#include "StackTensor3D.h"
#include "larcv/core/DataFormat/EventVoxel3D.h"
namespace larcv {

  static StackTensor3DProcessFactory __global_StackTensor3DProcessFactory__;

  StackTensor3D::StackTensor3D(const std::string name)
    : ProcessBase(name)
  {}

  void StackTensor3D::configure_labels(const PSet& cfg)
  {
    _tensor3d_producer_v.clear();
    _output_producer_v.clear();
    _tensor3d_producer_v = cfg.get<std::vector<std::string> >("Tensor3DProducerList", _tensor3d_producer_v);
    _output_producer_v = cfg.get<std::vector<std::string> >("OutputProducerList", _output_producer_v);
    _reference_producer = cfg.get<std::string>("ReferenceProducer");

    if (_tensor3d_producer_v.empty()) {
      auto tensor3d_producer = cfg.get<std::string>("Tensor3DProducer", "");
      auto output_producer    = cfg.get<std::string>("OutputProducer", "");
      if (!tensor3d_producer.empty()) {
        _tensor3d_producer_v.push_back(tensor3d_producer);
        _output_producer_v.push_back(output_producer);
      }
    }

    if (_output_producer_v.empty()) {
      _output_producer_v.resize(_tensor3d_producer_v.size(), "");
    }

    else if (_output_producer_v.size() != _tensor3d_producer_v.size()) {
      LARCV_CRITICAL() << "Tensor3DProducer and OutputProducer must have the same array length!" << std::endl;
      throw larbys();
    }

  }

  void StackTensor3D::configure(const PSet& cfg)
  {
    configure_labels(cfg);

    _pool_type_v = cfg.get<std::vector<unsigned short> >("PoolTypeList", _pool_type_v);
    if (_pool_type_v.empty()) {
      auto pool_type = cfg.get<unsigned short>("PoolType", (unsigned short)kSumPool);
      _pool_type_v.resize(_tensor3d_producer_v.size(), pool_type);
    } else if (_pool_type_v.size() != _tensor3d_producer_v.size()) {
      LARCV_CRITICAL() << "PoolTypeList size mismatch with other input parameters!" << std::endl;
      throw larbys();
    }

    _num_tensor = cfg.get<size_t>("NumTensor3D");
    _fuzzy_distance = cfg.get<double>("FuzzyDistance", -1.);
  }

  void StackTensor3D::initialize()
  {}

  bool StackTensor3D::process(IOManager& mgr)
  {
    auto const& ref_tensor3d = mgr.get_data<larcv::EventSparseTensor3D>(_reference_producer);
    auto const& ref_meta = ref_tensor3d.meta();

    for (size_t producer_index = 0; producer_index < _tensor3d_producer_v.size(); ++producer_index) {

      auto const& tensor3d_producer = _tensor3d_producer_v[producer_index];
      auto output_producer = _output_producer_v[producer_index];
      if (output_producer.empty()) output_producer = tensor3d_producer + "_compressed";
      auto const& _pool_type = _pool_type_v[producer_index];

      //larcv::VoxelSet vs;
      larcv::SparseTensor3D vs;
      vs.meta(ref_meta);

      for (size_t tensor_index = 0; tensor_index < _num_tensor; ++tensor_index) {

        auto const& ev_tensor3d = mgr.get_data<larcv::EventSparseTensor3D>(tensor3d_producer);

        switch(_pool_type) {
        case kSumPool:
	  for(auto const& vox : ev_tensor3d.as_vector())
            vs.add(vox);
          break;
        case kMaxPool:
        case kMinPool:
          for(auto const& vox : ev_tensor3d.as_vector()) {
            auto const& exist_vox = (_fuzzy_distance <= 0.) ? vs.find(vox.id()) : vs.close(vox.id(), _fuzzy_distance);
            if(exist_vox.id() == kINVALID_VOXELID) {
              vs.add(vox);
              continue;
            }
            if(_pool_type == kMaxPool && vox.value() > exist_vox.value())
              vs.emplace(vox.id(),vox.value(),false);
	    else if(_pool_type == kMinPool && vox.value() < exist_vox.value())
	      vs.emplace(vox.id(),vox.value(),false);
          }
        break;
        }
      }

      auto& out_tensor3d = mgr.get_data<larcv::EventSparseTensor3D>(output_producer);
      out_tensor3d.emplace(std::move(vs),ref_meta);
    }

    return true;
  }

  void StackTensor3D::finalize()
  {}

}
#endif
