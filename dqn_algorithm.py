from collections import namedtuple, deque
import torch
import numpy as np


def dqn(agent, env, n_episodes=2000, max_t=100, eps_start=1.0, eps_end=0.01,
        eps_decay=0.995, max_len=50):
    """Deep Q-Learning.
    
    Params
    ======
        agent (object): una instancia de un agente
        env (object): un entorno de OpenAI
        n_episodes (int): numero maximo de episodios de entrenamiento (n_episodios)
        max_t (int): numero maximo de pasos por episodio (n_entrenamiento)
        eps_start (float): valor inicial de epsilon
        eps_end (float): valor final de epsilon
        eps_decay (float): factor de multiplicacion (por episodio) de epsilon
        max_len (int): cantidad máxima de episodios a tomar en cuenta en el
                       cálculo de métricas
    """
    scores = []                             # puntuaciones de cada episodio
    scores_window = deque(maxlen=max_len)   # puntuaciones de los ultimos 100 episodios
    eps = eps_start                         # inicializar epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            
            # elegir accion At con politica e-greedy
            action = agent.act(state, eps)
            
            # aplicar At y obtener Rt+1, St+1
            next_state, reward, done, _ = env.step(action)
            
            # almacenar <St, At, Rt+1, St+1>
            agent.memory.add(state, action, reward, next_state, done)
            
            # train & update
            agent.step(state, action, reward, next_state, done)
            
            # avanzar estado
            state = next_state
            score += reward
            
            if done:
                break

        scores_window.append(score)       # guardar ultima puntuacion
        scores.append(score)              # guardar ultima puntuacion
        eps = max(eps_end, eps_decay*eps) # reducir epsilon
        
        print('\rEpisodio {}\tPuntuacion media (ultimos {:d}):{:.2f}'.format(i_episode, max_len, np.mean(scores_window)), end="")
        if i_episode % max_len == 0:
            print('\rEpisodio {}\tPuntuacion media ({:d} anteriores): {:.2f}'.format(i_episode, max_len, np.mean(scores_window)))
        if np.mean(scores_window)>=195.0:
            # Por qué 2? 
            print('\nProblema resuelto en {:d} episodios!\tPuntuacion media (ultimos {:d}): {:.2f}'.format(i_episode-max_len, max_len, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint-football.pth') # guardar pesos de agente entrenado
            break
    return scores



