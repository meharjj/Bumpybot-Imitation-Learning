from copy import deepcopy
import torch

## requirements
# env.step(actions) returns: next_obs, rewards, terms, truncs, infos 
# policy.get_action_and_value(obs) returns: actions, logprobs, etropy, values

## Device 
if torch.cuda.is_available():
    device = torch.device("cuda")
#elif torch.backends.mps.is_available(): #MPS is buggy
#    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('device is {}'.format(device))

def torch_obs(ob, device=device):
    ## TODO
    # write function to return tuple of torch obs from dict/tuple/list of *obs
    return ob

def collect_rollout(env, policy, n_steps, rollout, device=device):
    next_obs = env.reset()
    total_episodic_return = 0

    for step in range(0, n_steps):
        obs = torch_obs(next_obs, device)
        
        actions, logprobs, _, values = policy.get_action_and_value(obs.to(device))
        #print(actions)
        #rollout["obs_img"][step] = obs[0]
        rollout["obs_vec"][step] = obs
        rollout["actions"][step] = actions
        
        rollout["logprobs"][step] = logprobs
        rollout["values"][step] = values.flatten()

        next_obs, rewards, terms, truncs, infos = env.step(actions)

        rollout["rewards"][step] = rewards
        rollout["terms"][step] = float(terms)
        """ for k, v in rollout.items(): 
            print(v.size())
        assert False  """   
        total_episodic_return += rollout["rewards"][step].cpu()
        if terms:

        #if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
            end_step = step
            break
    else:
        end_step = step
    return rollout, total_episodic_return, end_step

def bootstrap_value(rollout, end_step, gamma=0.99, gae_lambda=0.95,device=device):
    rollout["advantages"] = torch.zeros_like(rollout["rewards"], device=device)
    value = rollout["values"][-1]
    for t in reversed(range(end_step)):
        if t == end_step-1:
            next_not_done = 1-rollout["terms"][-1]
            next_val = value
        else:
            next_not_done = 1-rollout["terms"][t + 1]
            next_val = rollout["values"][t + 1]
        delta = (
            rollout["rewards"][t]
            + gamma * next_val * next_not_done
            - rollout["values"][t]
            )
        rollout["advantages"][t] = delta + gamma * gae_lambda * next_not_done * rollout["advantages"][t + 1]
        rollout["returns"] = rollout["advantages"] + rollout["values"]
    return rollout

def batchify(end_step, rollout):
    #print(rollout["obs_vec"].size())
   
    #batched_rollout = {
    #    k : torch.flatten(v[:end_step], start_dim=0, end_dim=1) 
    #    for k,v in rollout.items()
    #    }
    #print(batched_rollout["obs_vec"].size())
    #assert False
    return rollout

def train(env, policy, optimizer, batch_size, epochs, end_step, rollout, clip_coef=0.3, ent_coef=0.0, vf_coef=0.5,max_grad_norm=0.5,device=device):
    updates = 0
    batched_rollout = batchify(end_step, rollout)
    
    clip_fracs = []
    for epoch in range(epochs):
        print("Epoch {}:".format(epoch))
       
        b_index = torch.randperm(batched_rollout["rewards"].size()[0]-1) + 1 #start at 1 so last_state indexing doesnt throw err
        for start in range(0, batched_rollout["obs_vec"].size()[0], batch_size):
            end = start + batch_size
            batch_index = b_index[start:end]

            _, newlogprob, entropy, value = policy.get_action_and_value(
                    (batched_rollout["obs_vec"][batch_index].to(device)), batched_rollout["actions"].long()[batch_index].to(device))
            logratio = newlogprob - batched_rollout["logprobs"][batch_index]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_fracs += [ ((ratio - 1.0).abs() > clip_coef).float().mean().item() ]

            # normalize advantaegs
            advantages = batched_rollout["advantages"][batch_index]
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

            # Policy loss
            pg_loss1 = -batched_rollout["advantages"][batch_index] * ratio
            pg_loss2 = -batched_rollout["advantages"][batch_index] * torch.clamp(
                ratio, 1 - clip_coef, 1 + clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            value = value.flatten()
            v_loss_unclipped = (value - batched_rollout["returns"][batch_index]) ** 2
            v_clipped = batched_rollout["values"][batch_index] + torch.clamp(
                value - batched_rollout["values"][batch_index],
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - batched_rollout["returns"][batch_index]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()

            loss = pg_loss - ent_coef * entropy_loss +  vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(policy.parameters(),max_grad_norm)
            optimizer.step()

            updates += batch_size

        print("Training Loss: {}".format(loss.item()))
        print("Clip Fraction: {}".format(sum(clip_fracs)/len(clip_fracs)))
    return updates
        
def PPO(env, policy, optimizer, max_steps, epochs, batch_size, n_steps=10_000,
    vec_obs_size=4, loss_coef=0.1, ent_coef=0.0, vf_coef=0.5, clip_coef=0.3, gamma=0.99, gae_lambda=0.95,max_grad_norm=0.5,device=device):
    total_steps = 0
    best_rewards = 0
    while total_steps < max_steps:
    
        # storage
        rollout = {}
        #rollout["obs_img"] = torch.zeros(n_steps, stack_size, *frame_size, device=device)
        rollout["obs_vec"]= torch.zeros(n_steps, vec_obs_size, device=device)
        rollout["actions"] = torch.zeros(n_steps,2, device=device)
        rollout["logprobs"] = torch.zeros(n_steps,2, device=device)
        rollout["rewards"] = torch.zeros(n_steps,1, device=device)
        rollout["terms"] = torch.zeros(n_steps,1, device=device)
        rollout["values"] = torch.zeros(n_steps,1, device=device)

        print("\n-------------------------------------------\n")
        print("Collectiing Rollout:")
        with torch.no_grad():
            rollout, total_episodic_return, end_step = collect_rollout(env, policy, n_steps, rollout, device)
            rollout = bootstrap_value(rollout, end_step, gamma, gae_lambda, device)
        
        if end_step < batch_size/10:
            print("Skipping Early Termination...\n")
            continue
        print("Episode Length: {}\n".format(end_step+1))
        
        print("Training for {} Epochs".format(epochs))
        total_steps += train(env, policy, optimizer, batch_size, epochs, end_step, rollout,max_grad_norm=max_grad_norm,device=device)
        if torch.sum(rollout["rewards"]).item()/end_step > best_rewards:
            print("saving new best model: {}".format(torch.sum(rollout["rewards"]).item()/end_step))
            best_rewards = torch.sum(rollout["rewards"]).item()/end_step
            best_model = deepcopy(policy)
            best_model_scripted = torch.jit.script(best_model)
            best_model_scripted.save("rl_best_model.pt")

        print("\n...{}/{}".format(total_steps,max_steps))

        model_scripted = torch.jit.script(policy) # export to TorchScript
        model_scripted.save("rl_model.pt") #save TorchScript model

