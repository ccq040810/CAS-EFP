import unittest
import torch
from scripts.run_dnn_dlinear_external import DLinear,masked_mse

class BaselineTests(unittest.TestCase):
    def test_dlinear_constant_decomposition_and_gradients(self):
        model=DLinear();x=torch.full((3,168),2.)
        trend=model.avg(torch.cat([x[:,:1].repeat(1,12),x,x[:,-1:].repeat(1,12)],1).unsqueeze(1)).squeeze(1)
        torch.testing.assert_close(trend,x);torch.testing.assert_close(model(x),model.seasonal(torch.zeros_like(x))+model.trend(x))
        model(x).sum().backward();self.assertTrue(all(p.grad is not None for p in model.parameters()))
    def test_missing_horizon_targets_do_not_affect_training(self):
        p=torch.tensor([[1.,1000.]],requires_grad=True);t=torch.zeros_like(p);mask=torch.tensor([[1.,0.]])
        loss=masked_mse(p,t,mask);self.assertEqual(loss.item(),1.);loss.backward();self.assertEqual(p.grad[0,1].item(),0.)

if __name__=='__main__':unittest.main()
