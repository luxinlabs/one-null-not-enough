import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import ABLATION_POLICIES,ABLATION_SCENARIOS,ACTIONS,POLICIES,TRUTHS,WORLDS,choose,expected_gain,mean,posterior_update,run,truth_mean
import numpy as np

def world_seed(target,start=0):
    return next(s for s in range(start,10_000) if run(s,'replicate','unidentifiable')['world']==target)

class Tests(unittest.TestCase):
 def test_ambiguous_pair(self):
  for a in ('ordinary','replicate'):self.assertEqual(mean(0,a),mean(1,a))
  self.assertTrue(any(mean(0,a)!=mean(1,a) for a in ACTIONS))
 def test_posterior(self):
  p=posterior_update(np.ones(4)/4,'calibrate',0.,.1)
  self.assertAlmostEqual(float(p.sum()),1.)
 def test_all_worlds_sampled(self):
  self.assertEqual({run(s,'replicate','unidentifiable')['world'] for s in range(100)},set(range(4)))
 def test_naive_falsely_rejects_masked_effect(self):
  # World 1 (effect present, instrument faulty): fault-unaware updating rejects the true H.
  r=run(world_seed(1),'scientific_only')
  self.assertEqual(r['false_rejection'],1)
 def test_joint_model_does_not_reject_masked_effect(self):
  r=run(world_seed(1),'replicate')
  self.assertEqual((r['false_rejection'],r['abstain']),(0,1))
 def test_common_random_numbers(self):
  # The first use of an action yields the same measurement under every policy for a seed.
  first={}
  for pol in POLICIES:
   seen=set()
   for step in run(3,pol)['history']:
    if step['action'] in seen:continue
    seen.add(step['action']);first.setdefault(step['action'],set()).add(round(step['measurement'],12))
  for a,vals in first.items():self.assertEqual(len(vals),1,a)
  self.assertEqual({run(3,p)['world'] for p in POLICIES},{run(3,'joint_risk')['world']})
 def test_identifiability_measures_before_abstaining(self):
  # World 2 is separable from world 0 by ordinary readings, so the policy must not abstain before acting.
  self.assertEqual(choose('identifiability',np.ones(4)/4,.25,np.random.default_rng(0),['ordinary','replicate']),'ordinary')
  r=run(0,'identifiability','unidentifiable')
  self.assertGreater(len(r['history']),0)
 def test_identifiability_abstains_when_unseparable(self):
  p=np.array([.5,.5,0.,0.]) # only the observationally equivalent pair remains
  self.assertIsNone(choose('identifiability',p,.25,np.random.default_rng(0),['ordinary','replicate']))
 def test_calibrate_first_reads_h(self):
  acts={s['action'] for s in run(0,'calibrate_first')['history']}
  self.assertIn('calibrate',acts);self.assertTrue(acts&{'ordinary','alternate_sensor'})
 def test_budget_limited_differs(self):
  self.assertLessEqual(run(0,'joint_risk','budget_limited')['spent'],2.+1e-9)
 def test_unidentifiable_never_resolves_masked_pair(self):
  for pol in POLICIES[1:]:
   self.assertEqual(run(world_seed(0),pol,'unidentifiable')['correct'],0,pol)
 def test_vectorized_gain_matches_closed_form(self):
  # Calibrating from a uniform prior reveals f but not H: zero expected risk reduction.
  self.assertAlmostEqual(expected_gain(np.ones(4)/4,'calibrate',.5,np.random.default_rng(0),512,'risk'),0.,places=6)
  self.assertGreater(expected_gain(np.ones(4)/4,'alternate_sensor',.25,np.random.default_rng(0),512,'risk'),.4)
 def test_specified_truth_matches_planning_model(self):
  for w,(h,f) in enumerate(WORLDS):
   for a in ACTIONS:self.assertEqual(truth_mean(h,f,a,3,'specified',.9),mean(w,a))
 def test_misspecified_truths_differ_from_model(self):
  self.assertNotEqual(truth_mean(0,0,'ordinary',5,'drift',0.),mean(0,'ordinary'))
  self.assertEqual(truth_mean(1,1,'ordinary',0,'intermittent',.9),1.) # fault absent on this measurement
  self.assertEqual(truth_mean(1,1,'positive_control',0,'imperfect_control',.1),1.) # false pass
  self.assertEqual(truth_mean(1,1,'calibrate',0,'unknown_fault',.5),0.) # undetected by calibration
 def test_all_truths_policies_scenarios_run(self):
  for truth in TRUTHS:
   for pol in POLICIES+ABLATION_POLICIES:
    for sc in ('resolvable',)+tuple(ABLATION_SCENARIOS):
     r=run(1,pol,sc,truth=truth,n_mc=16);self.assertIn(r['abstain'],(0,1))
  self.assertNotIn('calibrate',{s['action'] for s in run(1,'joint_risk','no_calibrate')['history']})
 def test_rejection_guard(self):
  # Low fault prior: unidentifiable ordinary readings push P(H) below 0.1. The guard abstains instead of rejecting.
  s=world_seed(1)
  self.assertEqual(run(s,'identifiability_stop','unidentifiable',prior_fault=.05,budget=10.)['false_rejection'],1)
  r=run(s,'identifiability','unidentifiable',prior_fault=.05,budget=10.)
  self.assertEqual((r['false_rejection'],r['abstain']),(0,1))
  self.assertEqual(run(world_seed(2),'identifiability')['correct'],1) # guard does not block resolvable decisions
 def test_robust_bayes(self):
  s=world_seed(1) # masked effect, unidentifiable, planner prior says faults are rare
  self.assertEqual(run(s,'joint_risk_voi','unidentifiable',prior_fault=.05,budget=10.)['false_rejection'],1)
  self.assertEqual(run(s,'robust_bayes','unidentifiable',prior_fault=.05,budget=10.)['abstain'],1)
  self.assertEqual(run(world_seed(2),'robust_bayes')['correct'],1) # decides when the evidence is strong
 def test_oracle_knows_fault(self):
  r=run(world_seed(1),'oracle_fault');self.assertEqual(r['correct'],1)
if __name__=='__main__':unittest.main()
