#pragma once

#include <vector>
#include <span>
#include <memory>
#include "Activation.h"

using data_pair = std::pair<Math::Matrix, Math::Matrix>;
using dataset = std::vector<data_pair>;
using dataset_span = std::span<data_pair>;
using dataset_span_const = std::span<const data_pair>;

using shared_afn_ptr = std::shared_ptr<Activation::ActivationFn>;
