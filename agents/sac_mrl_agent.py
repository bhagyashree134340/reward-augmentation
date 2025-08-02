import torch
import wandb
import torch.nn.functional as F
from agents.sac_agent import SACAgent
from utils.polyak import polyak_update


class SACMRLAgent(SACAgent):
    def __init__(self, env, eval_env, alpha=0.9, tau_m=0.03, lo=-1.0, encoder=None, encoder_output_dim=None, **kwargs):
        super().__init__(env, eval_env, encoder=encoder, encoder_output_dim=encoder_output_dim, **kwargs)
        self.alpha_m = alpha
        self.tau_m = tau_m
        self.lo = lo

    def update(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, current_step):
        not_done = 1 - done_batch.unsqueeze(-1).float()
        ent_coef = self.log_ent_coef.exp()

        wandb.log({
            "action/mean": act_batch.mean().item(),
            "action/std": act_batch.std().item(),
        }, step=current_step)

        with torch.no_grad():
            _, log_pi = self.actor(obs_batch)

            action_log_probs = log_pi.sum(dim=-1, keepdim=True)
            munchausen_term = torch.clamp(self.tau_m * action_log_probs, min=self.lo, max=0.0)

            r_mun = rew_batch.unsqueeze(-1) + self.alpha_m * munchausen_term

        wandb.log({
            "log_pi": action_log_probs.mean().item(),
            "r_mun": r_mun.mean().item(),
            "bonus_term": munchausen_term.mean().item()
        }, step=current_step)

        with torch.no_grad():
            next_action, next_log_prob = self.actor_target(next_obs_batch)
            q1_target_val = self.q1_target(next_obs_batch, next_action)
            q2_target_val = self.q2_target(next_obs_batch, next_action)
            q_target_min = torch.min(q1_target_val, q2_target_val)

            q_target = r_mun + self.gamma * not_done * (
                    q_target_min - ent_coef * next_log_prob.sum(dim=-1, keepdim=True)
            )

        q_losses = []
        for q_func, q_opt in [(self.q1, self.q1_optimizer), (self.q2, self.q2_optimizer)]:
            q_pred = q_func(obs_batch, act_batch)
            q_loss = F.mse_loss(q_pred, q_target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()
            q_losses.append(q_loss.item())

        new_action, new_log_prob = self.actor(obs_batch)
        q1_new = self.q1(obs_batch, new_action)
        q2_new = self.q2(obs_batch, new_action)
        min_q = torch.min(q1_new, q2_new)

        actor_loss = (-min_q + ent_coef * new_log_prob.sum(dim=-1, keepdim=True)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ent_coef_loss = -(self.log_ent_coef.exp() * (new_log_prob.detach() + self.target_entropy)).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        polyak_update(self.q1.parameters(), self.q1_target.parameters(), self.tau)
        polyak_update(self.q2.parameters(), self.q2_target.parameters(), self.tau)
        polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        wandb.log({
            "loss/q1_loss": q_losses[0],
            "loss/q2_loss": q_losses[1],
            "loss/actor_loss": actor_loss.item(),
            "loss/ent_coef_loss": ent_coef_loss.item(),
            "policy/entropy": -new_log_prob.mean().item(),
            "ent_coef/value": ent_coef.item(),
        }, step=current_step)
