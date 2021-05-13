#ifndef __SmearTensor3D_CXX__
#define __SmearTensor3D_CXX__

#include "SmearTensor3D.h"
#include "larcv/core/DataFormat/EventVoxel3D.h"
namespace larcv {

  static SmearTensor3DProcessFactory __global_SmearTensor3DProcessFactory__;

  SmearTensor3D::SmearTensor3D(const std::string name)
    : ProcessBase(name)
  {}

  void SmearTensor3D::configure_labels(const PSet& cfg)
  {
    _target_producer_v.clear();
    _output_producer_v.clear();
    _target_producer_v = cfg.get<std::vector<std::string> >("TargetProducerList", _target_producer_v);
    _output_producer_v   = cfg.get<std::vector<std::string> >("OutputProducerList", _output_producer_v);

    if (_target_producer_v.empty()) {
      auto tensor3d_producer = cfg.get<std::string>("TargetProducer", "");
      auto output_producer   = cfg.get<std::string>("OutputProducer", "");
      if (!tensor3d_producer.empty()) {
        _target_producer_v.push_back(tensor3d_producer);
        _output_producer_v.push_back(output_producer);
      }
    }

    if (_output_producer_v.empty()) {
      _output_producer_v.resize(_target_producer_v.size(), "");
    }
    else if (_output_producer_v.size() != _target_producer_v.size()) {
      LARCV_CRITICAL() << "TargetProducer and OutputProducer must have the same array length!" << std::endl;
      throw larbys();
    }
  }

  void SmearTensor3D::configure(const PSet& cfg)
  {
    configure_labels(cfg);

    _voxel_smear_v = cfg.get<std::vector<float> >("SmearList", _voxel_smear_v);
    if (_voxel_smear_v.empty()) {
      auto voxel_smear = cfg.get<float>("Smear", 0.);
      _voxel_smear_v.resize(_target_producer_v.size(), voxel_smear);
    } else if (_voxel_smear_v.size() != _target_producer_v.size()) {
      LARCV_CRITICAL() << "SmearList size mismatch with other input parameters!" << std::endl;
      throw larbys();
    }
  }

  void SmearTensor3D::initialize()
  {}

  bool SmearTensor3D::process(IOManager& mgr)
  {
    for (size_t producer_index = 0; producer_index < _target_producer_v.size(); ++producer_index) {

      auto const& target_producer = _target_producer_v[producer_index];
      auto output_producer = _output_producer_v[producer_index];
      if (output_producer.empty()) output_producer = target_producer + "_smear";

      auto const& ev_tensor3d = mgr.get_data<larcv::EventSparseTensor3D>(target_producer);
      auto& ev_output = mgr.get_data<larcv::EventSparseTensor3D>(output_producer);

      if (ev_output.meta().valid()) {
        static bool one_time_warning = true;
        if (_output_producer_v[producer_index].empty()) {
          LARCV_CRITICAL() << "Over-writing existing EventSparseTensor3D data for label "
                           << output_producer << std::endl;
          throw larbys();
        }
        else if (one_time_warning) {
          LARCV_WARNING() << "Output EventSparseTensor3D producer " << output_producer
                          << " already holding valid data will be over-written!" << std::endl;
          one_time_warning = false;
        }
      }

      auto const& meta = ev_tensor3d.meta();
      auto const& voxel_smear = _voxel_smear_v[producer_index];
      std::default_random_engine generator;
      std::normal_distribution<float> distribution(0.0,voxel_smear);
      larcv::VoxelSet res_data;
      for (auto const& vox : ev_tensor3d.as_vector()) {
        LARCV_DEBUG() << "Smearing vox ID " << vox.id() << " charge " << vox.value() << std::endl;
        float charge = vox.value() + distribution(generator);
        LARCV_DEBUG() << "... to charge " << charge << std::endl;
        res_data.emplace(vox.id(), charge, true);
      }

      ev_output.emplace(std::move(res_data), meta);
    }

   return true;
  }

  void SmearTensor3D::finalize()
  {}

}
#endif
