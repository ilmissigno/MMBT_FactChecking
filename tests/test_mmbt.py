"""
MMBT Fact-Checking - Unit Tests
================================

Run tests with: pytest tests/ -v
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTransforms:
    """Test image transformation functions"""
    
    def test_gaussian_noise_shape(self):
        """Test that Gaussian noise preserves tensor shape"""
        from mmbt.data.transforms import AddGaussianNoise
        
        noise_fn = AddGaussianNoise(std=0.1)
        tensor = torch.rand(3, 224, 224)
        
        noisy = noise_fn(tensor)
        
        assert noisy.shape == tensor.shape
    
    def test_gaussian_noise_range(self):
        """Test that output is clamped to [0, 1]"""
        from mmbt.data.transforms import AddGaussianNoise
        
        noise_fn = AddGaussianNoise(std=0.5)  # High noise
        tensor = torch.rand(3, 224, 224)
        
        noisy = noise_fn(tensor)
        
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0
    
    def test_different_noise_levels(self):
        """Test that higher noise levels produce more variation"""
        from mmbt.data.transforms import AddGaussianNoise
        
        tensor = torch.rand(3, 224, 224)
        
        low_noise = AddGaussianNoise(std=0.01)
        high_noise = AddGaussianNoise(std=0.3)
        
        low_result = low_noise(tensor.clone())
        high_result = high_noise(tensor.clone())
        
        low_diff = (low_result - tensor).abs().mean()
        high_diff = (high_result - tensor).abs().mean()
        
        assert high_diff > low_diff
    
    def test_salt_pepper_noise(self):
        """Test salt and pepper noise"""
        from mmbt.data.transforms import AddSaltPepperNoise
        
        noise_fn = AddSaltPepperNoise(amount=0.1)
        tensor = torch.rand(3, 224, 224) * 0.5 + 0.25  # Values around 0.5
        
        noisy = noise_fn(tensor)
        
        # Should have some 0s and 1s
        assert (noisy == 0.0).any() or (noisy == 1.0).any()
    
    def test_get_transforms_basic(self):
        """Test basic transform creation"""
        from mmbt.data.transforms import get_transforms
        from types import SimpleNamespace
        
        args = SimpleNamespace()
        transforms = get_transforms(args)
        
        assert transforms is not None
    
    def test_get_transforms_with_noise(self):
        """Test transform creation with noise"""
        from mmbt.data.transforms import get_transforms
        from types import SimpleNamespace
        from PIL import Image
        
        args = SimpleNamespace()
        
        # Create a dummy image
        img = Image.new('RGB', (256, 256), color='red')
        
        # Without noise
        transforms_clean = get_transforms(args, noise_level=0.0)
        result_clean = transforms_clean(img)
        
        # With noise
        transforms_noisy = get_transforms(args, noise_level=0.3)
        result_noisy = transforms_noisy(img)
        
        assert result_clean.shape == result_noisy.shape


class TestComplexityBenchmark:
    """Test complexity benchmarking module"""
    
    def test_timer_context_manager(self):
        """Test Timer utility"""
        from mmbt.benchmarks.complexity import Timer
        import time
        
        with Timer("test") as t:
            time.sleep(0.1)
        
        assert t.elapsed >= 0.1
        assert t.elapsed < 0.2
    
    def test_memory_usage(self):
        """Test memory usage function"""
        from mmbt.benchmarks.complexity import get_memory_usage
        
        memory = get_memory_usage()
        
        assert 'rss_mb' in memory
        assert memory['rss_mb'] > 0
    
    @pytest.mark.skipif(
        not Path("/usr/bin/faiss").exists() and True,  # Skip if FAISS not installed
        reason="FAISS not installed"
    )
    def test_faiss_benchmark(self):
        """Test FAISS benchmark"""
        try:
            from mmbt.benchmarks.complexity import FAISSBenchmark
            
            bench = FAISSBenchmark()
            result = bench.benchmark_flat_index(n_vectors=100, dim=32)
            
            assert result.name == "FlatL2"
            assert 'index_build_time' in result.times
        except ImportError:
            pytest.skip("FAISS not available")


class TestMetrics:
    """Test ranking metrics"""
    
    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking"""
        from mmbt.metrics.RankMetrics import ndcg
        
        # Perfect ranking: prediction matches truth
        y_pred = torch.tensor([3.0, 2.0, 1.0, 0.0])
        y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])  # 2 relevant items at top
        
        result = ndcg(y_pred, y_true, ats=[2, 4])
        
        # Should be close to 1 for perfect ranking
        assert result[0] > 0.9  # NDCG@2
    
    def test_hit_ratio(self):
        """Test hit ratio calculation"""
        from mmbt.metrics.RankMetrics import getHitRatioForList
        
        ranked_list = [1, 2, 3, 4, 5]
        ground_truth = {3, 7, 9}  # Item 3 is in ranked list
        
        hit = getHitRatioForList(ranked_list, ground_truth)
        
        assert hit == 1.0
    
    def test_hit_ratio_no_match(self):
        """Test hit ratio with no matches"""
        from mmbt.metrics.RankMetrics import getHitRatioForList
        
        ranked_list = [1, 2, 4, 5, 6]
        ground_truth = {3, 7, 9}  # No overlap
        
        hit = getHitRatioForList(ranked_list, ground_truth)
        
        assert hit == 0.0


class TestDataLoading:
    """Test data loading functions"""
    
    def test_collate_function_basic(self):
        """Test that collate function runs without error"""
        # This is a basic smoke test - actual testing would need mock data
        pass
    
    def test_vocab_creation(self):
        """Test vocabulary creation"""
        from mmbt.data.vocab import Vocab
        
        vocab = Vocab()
        
        assert hasattr(vocab, 'stoi')
        assert hasattr(vocab, 'itos')


class TestModels:
    """Test model components"""
    
    def test_image_encoder_output_shape(self):
        """Test ImageEncoder output shape"""
        from mmbt.models.image import ImageEncoder
        from types import SimpleNamespace
        
        args = SimpleNamespace(
            image_model='resnet50',
            num_image_embeds=3,
            img_embed_pool_type='avg'
        )
        
        encoder = ImageEncoder(args)
        
        # Input: batch of images
        x = torch.rand(2, 3, 224, 224)
        
        output = encoder(x)
        
        # Output should be [batch, num_embeds, 2048]
        assert output.shape == (2, 3, 2048)


class TestLoss:
    """Test loss functions"""
    
    def test_cross_similarity_shape(self):
        """Test CrossSimilarity output shape"""
        from mmbt.losses.CrossSimilarity import CrossSimilarity
        
        loss_fn = CrossSimilarity()
        
        query = torch.rand(8, 128)
        doc = torch.rand(8, 128)
        
        similarity = loss_fn.similarities(query, doc)
        
        # Should output similarity score per sample
        assert similarity.shape[1] == 8


# Integration tests
class TestIntegration:
    """Integration tests (slower, require more setup)"""
    
    @pytest.mark.slow
    def test_full_forward_pass(self):
        """Test complete forward pass through model"""
        # Would require full model setup
        pass
    
    @pytest.mark.slow
    def test_data_pipeline(self):
        """Test data loading pipeline"""
        # Would require actual data files
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
