#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#include <vector>
#include <string>
#include <memory>
#include "status_code.hpp"
#include "runtime_datatype.hpp"
#include "data/tensor.hpp"

namespace kuiper_infer{
    struct RuntimeOperand{
        std::string name; // 操作数名称
        std::vector<int32_t> shapes;  // 操作数形状
        std::vector<std::shared_ptr<Tensor<float>>> datas; // 存储操作数
        RuntimeDataType type = RuntimeDataType::kTypeUnknown; // 操作数类型, 一般是float
    };
}

#endif