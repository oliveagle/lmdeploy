// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <mutex>
#include <stack>
#include <functional>

#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

class TensorMapPool {
public:
    static TensorMapPool& Instance()
    {
        static TensorMapPool pool;
        return pool;
    }

    struct PooledTensorMap {
        TensorMap map;
        std::function<void()> on_return;

        PooledTensorMap() = default;
        ~PooledTensorMap()
        {
            if (on_return) {
                on_return();
            }
        }

        PooledTensorMap(const PooledTensorMap&) = delete;
        PooledTensorMap& operator=(const PooledTensorMap&) = delete;

        PooledTensorMap(PooledTensorMap&& other) noexcept
            : map(std::move(other.map)), on_return(std::move(other.on_return))
        {
            other.on_return = nullptr;
        }

        PooledTensorMap& operator=(PooledTensorMap&& other) noexcept
        {
            if (this != &other) {
                if (on_return) {
                    on_return();
                }
                map = std::move(other.map);
                on_return = std::move(other.on_return);
                other.on_return = nullptr;
            }
            return *this;
        }
    };

    std::shared_ptr<PooledTensorMap> Acquire()
    {
        std::lock_guard lock{mutex_};

        if (pool_.empty()) {
            auto pooled = std::make_shared<PooledTensorMap>();
            pooled->on_return = [this, p = pooled.get()]() {
                Return(p);
            };
            return pooled;
        }

        auto* pooled = pool_.top();
        pool_.pop();

        auto shared = std::shared_ptr<PooledTensorMap>(pooled, [this](PooledTensorMap* p) {
            Return(p);
        });

        shared->on_return = [this, p = shared.get()]() {
            Return(p);
        };

        return shared;
    }

    void SetMaxPoolSize(size_t max_size)
    {
        std::lock_guard lock{mutex_};
        max_pool_size_ = max_size;
    }

    size_t GetPoolSize() const
    {
        std::lock_guard lock{mutex_};
        return pool_.size();
    }

    void Clear()
    {
        std::lock_guard lock{mutex_};
        while (!pool_.empty()) {
            delete pool_.top();
            pool_.pop();
        }
    }

private:
    TensorMapPool() = default;

    ~TensorMapPool()
    {
        Clear();
    }

    TensorMapPool(const TensorMapPool&) = delete;
    TensorMapPool& operator=(const TensorMapPool&) = delete;

    void Return(PooledTensorMap* pooled)
    {
        if (!pooled) {
            return;
        }

        std::lock_guard lock{mutex_};

        if (pool_.size() >= max_pool_size_) {
            delete pooled;
            return;
        }

        pooled->map.clear();
        pool_.push(pooled);
    }

private:
    mutable std::mutex mutex_;
    std::stack<PooledTensorMap*> pool_;
    size_t max_pool_size_ = 128;
};

inline std::shared_ptr<TensorMapPool::PooledTensorMap> AcquireTensorMap()
{
    return TensorMapPool::Instance().Acquire();
}

}  // namespace turbomind::core
