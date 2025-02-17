#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>

namespace kuiper_infer {

	Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
		data_ = arma::fcube(rows, cols, channels);  // Ϊʲô��ʼ��˳�򷴹���
	}

	Tensor<float>::Tensor(const Tensor& tensor) {
		this->data_ = tensor.data_;
		this->raw_shapes_ = tensor.raw_shapes_;
	}

	Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
		if (this != &tensor) {
			this->data_ = tensor.data_;
			this->raw_shapes_ = tensor.raw_shapes_;
		}
		return *this;
	}

	uint32_t Tensor<float>::rows() const {
		CHECK(!this->data_.empty());
		return this->data_.n_rows;
	}

	uint32_t Tensor<float>::cols() const {
		CHECK(!this->data_.empty());
		return this->data_.n_cols;
	}

	uint32_t Tensor<float>::channels() const {
		CHECK(!this->data_.empty());
		return this->data_.n_slices;
	}

	uint32_t Tensor<float>::size() const {
		CHECK(!this->data_.empty());
		return this->data_.size();
	}

	void Tensor<float>::set_data(const arma::fcube& data) {
		CHECK(data.n_rows == this->data_.n_rows) << data.n_rows << " != " << this->data_.n_rows;
		CHECK(data.n_cols == this->data_.n_cols) << data.n_cols << " != " << this->data_.n_cols;
		CHECK(data.n_slices == this->data_.n_slices) << data.n_slices << " != " << this->data_.n_slices;
		this->data_ = data;
	}

	bool Tensor<float>::empty() const {
		return this->data_.empty();
	}

	float Tensor<float>::index(uint32_t offset) const {
		CHECK(offset < this->data_.size());
		return this->data_.at(offset);
	}

	std::vector<uint32_t> Tensor<float>::shapes() const {
		CHECK(!this->data_.empty());
		return { this->channels(), this->rows(), this->cols() };
	}

	arma::fcube& Tensor<float>::data() {
		return this->data_;
	}

	const arma::fcube& Tensor<float>::data() const {
		return this->data_;
	}

	arma::fmat& Tensor<float>::at(uint32_t channel) {
		CHECK_LT(channel, this->channels());
		return this->data_.slice(channel); 
	}

	const arma::fmat& Tensor<float>::at(uint32_t channel) const {
		CHECK_LT(channel, this->channels());
		return this->data_.slice(channel);
	}

	float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
		CHECK_LT(row, this->rows()); 
		CHECK_LT(col, this->cols());
		CHECK_LT(channel, this->channels());
		return this->data_.at(row, col, channel);
	}

	float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
		CHECK_LT(row, this->rows());
		CHECK_LT(col, this->cols());
		CHECK_LT(channel, this->channels());
		return this->data_.at(row, col, channel);
	}

	void Tensor<float>::Padding(const std::vector<uint32_t>& pads, float padding_value) {
		CHECK(!this->data_.empty());
		CHECK_EQ(pads.size(), 4);
		uint32_t pad_rows1 = pads.at(0);  // up
		uint32_t pad_rows2 = pads.at(1);  // bottom
		uint32_t pad_cols1 = pads.at(2);  // left
		uint32_t pad_cols2 = pads.at(3);  // right

		//this->data_.insert_rows(0, pad_rows1);
		//this->data_.insert_rows(this->data_.n_rows, pad_rows2);
		//this->data_.insert_cols(0, pad_cols1);
		//this->data_.insert_cols(this->data_.n_cols, pad_cols2);
		//this->raw_shapes_ = this->shapes();

		arma::fcube new_data(this->data_.n_rows + pad_rows1 + pad_rows2,
			this->data_.n_cols + pad_cols1 + pad_cols2,
			this->data_.n_slices);
		new_data.fill(padding_value);

		new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows2 - 1,
			new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) =
			this->data_;
		this->data_ = std::move(new_data);
	}

	void Tensor<float>::Fill(float value) {
		CHECK(!this->data_.empty());
		this->data_.fill(value);
	}

	void Tensor<float>::Fill(const std::vector<float>& values) {
		CHECK(!this->data_.empty());
		const uint32_t total_elems = this->data_.size();
		CHECK_EQ(values.size(), total_elems);

		const uint32_t rows = this->rows();
		const uint32_t cols = this->cols();
		const uint32_t planes = rows * cols;
		const uint32_t channels = this->data_.n_slices;

		//todo 
		for (uint32_t i = 0; i < channels; ++i) {
			auto& channel_data = this->data_.slice(i);
			const arma::fmat& channel_data_t = arma::fmat(values.data() + i * planes, this->cols(), this->rows());
			channel_data = channel_data_t.t();
		}

	}

	void Tensor<float>::Show() {
		for (uint32_t i = 0; i < this->channels(); ++i) {
			LOG(INFO) << "Channel: " << i;
			LOG(INFO) << "\n" << this->data_.slice(i);
		}
	}


	void Tensor<float>::Flatten() {
		CHECK(!this->data_.empty());
		const uint32_t size = this->data_.size();
		arma::fcube linear_cube(size, 1, 1);

		uint32_t channel = this->channels();
		uint32_t rows = this->rows();
		uint32_t cols = this->cols();
		uint32_t index = 0;

		for (uint32_t c = 0; c < channel; ++c) {
			const arma::fmat& matrix = this->data_.slice(c);

			for (uint32_t r = 0; r < rows; ++r) {
				for (uint32_t c_ = 0; c_ < cols; ++c_) {
					linear_cube.at(index, 0, 0) = matrix.at(r, c_);
					index += 1;
				}
			}
		}
		CHECK_EQ(index, size);
		this->data_ = linear_cube;
		this->raw_shapes_ = std::vector<uint32_t>{ size };
	}

	void Tensor<float>::Rand() {
		CHECK(!this->data_.empty());
		this->data_.randn();
	}

	void Tensor<float>::Ones() {
		CHECK(!this->data_.empty());
		this->data_.fill(1.);
	}

}