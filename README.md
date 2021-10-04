# Project 2: Continous Control
<h3>Introduction</h3>
For this project, DDPG and DDPG with priority experience replay were developed to train and evaluate a double-joined arm agent to follow target in the unity ML-agent Reacher environment.  This project is part of the Deep Reinforcement Learning Nanodegree program.

<img class="image--26lOQ" src="https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif" alt="Unity ML-Agents Reacher Environment" width="500px">

<h3>Rewards</h3>
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of DDPG and DDPG_PER agents is to maintain its position at the target location for as many time steps as possible.

<h3>Environment</h3>
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

# Getting Started
<h3>Download DRLND repository</h3>
<p>To set up your python environment to run the code in this repository, follow the instructions below.</p>
<p>Create (and activate) a new environment with Python 3.6.</p>
<ol>
<li>
<p>Create (and activate) a new environment with Python 3.6.</p>
<ul>
<li><strong>Linux</strong> or <strong>Mac</strong>:</li>
</ul>
<div class="highlight highlight-source-shell position-relative"><pre>conda create --name drlnd python=3.6
<span class="pl-c1">source</span> activate drlnd</pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="conda create --name drlnd python=3.6
source activate drlnd
" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-paste js-clipboard-clippy-icon m-2">
    <path fill-rule="evenodd" d="M5.75 1a.75.75 0 00-.75.75v3c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-3a.75.75 0 00-.75-.75h-4.5zm.75 3V2.5h3V4h-3zm-2.874-.467a.75.75 0 00-.752-1.298A1.75 1.75 0 002 3.75v9.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 13.25v-9.5a1.75 1.75 0 00-.874-1.515.75.75 0 10-.752 1.298.25.25 0 01.126.217v9.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.126-.217z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-text-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<ul>
<li><strong>Windows</strong>:</li>
</ul>
<div class="highlight highlight-source-shell position-relative"><pre>conda create --name drlnd python=3.6 
activate drlnd</pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="conda create --name drlnd python=3.6 
activate drlnd
" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-paste js-clipboard-clippy-icon m-2">
    <path fill-rule="evenodd" d="M5.75 1a.75.75 0 00-.75.75v3c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-3a.75.75 0 00-.75-.75h-4.5zm.75 3V2.5h3V4h-3zm-2.874-.467a.75.75 0 00-.752-1.298A1.75 1.75 0 002 3.75v9.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 13.25v-9.5a1.75 1.75 0 00-.874-1.515.75.75 0 10-.752 1.298.25.25 0 01.126.217v9.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.126-.217z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-text-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
</li>

<li>
<p>Clone the repository (if you haven't already!), and navigate to the <code>python/</code> folder.  Then, install several dependencies.</p>
</li>
</ol>
<div class="highlight highlight-source-shell position-relative"><pre>git clone https://github.com/udacity/deep-reinforcement-learning.git
<span class="pl-c1">cd</span> deep-reinforcement-learning/python
pip install <span class="pl-c1">.</span></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-paste js-clipboard-clippy-icon m-2">
    <path fill-rule="evenodd" d="M5.75 1a.75.75 0 00-.75.75v3c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-3a.75.75 0 00-.75-.75h-4.5zm.75 3V2.5h3V4h-3zm-2.874-.467a.75.75 0 00-.752-1.298A1.75 1.75 0 002 3.75v9.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 13.25v-9.5a1.75 1.75 0 00-.874-1.515.75.75 0 10-.752 1.298.25.25 0 01.126.217v9.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.126-.217z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-text-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
  <ol start="4">
<li>Create an <a href="http://ipython.readthedocs.io/en/stable/install/kernel_install.html" rel="nofollow">IPython kernel</a> for the <code>drlnd</code> environment.</li>
</ol>
<div class="highlight highlight-source-shell position-relative"><pre>python -m ipykernel install --user --name drlnd --display-name <span class="pl-s"><span class="pl-pds">"</span>drlnd<span class="pl-pds">"</span></span></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="python -m ipykernel install --user --name drlnd --display-name &quot;drlnd&quot;
" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-paste js-clipboard-clippy-icon m-2">
    <path fill-rule="evenodd" d="M5.75 1a.75.75 0 00-.75.75v3c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-3a.75.75 0 00-.75-.75h-4.5zm.75 3V2.5h3V4h-3zm-2.874-.467a.75.75 0 00-.752-1.298A1.75 1.75 0 002 3.75v9.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 13.25v-9.5a1.75 1.75 0 00-.874-1.515.75.75 0 10-.752 1.298.25.25 0 01.126.217v9.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.126-.217z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-text-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
  <ol start="5">
<li>Before running code in a notebook, change the kernel to match the <code>drlnd</code> environment by using the drop-down <code>Kernel</code> menu.</li>
</ol>
<p><a target="_blank" rel="noopener noreferrer" href="https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png"><img src="https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png" alt="Kernel" title="Kernel" style="max-width:100%;"></a></p>


<h3>Download Unity Environment</h3>
Download the environment from one of the links below. You need only select the environment that matches your operating system:

<h3 id="version-1-one-1-agent">Version 1: One (1) Agent</h3>

<ul>
<li>Linux: <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip">click here</a></li>
<li>Mac OSX: <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip">click here</a></li>
<li>Windows (32-bit): <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip">click here</a></li>
<li>Windows (64-bit): <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip">click here</a></li>
</ul>

<p>Then, place the file in the <code>p2_continuous-control/</code> folder in the DRLND GitHub repository, and unzip (or decompress) the file.</p>

<p>(<em>For Windows users</em>) Check out <a target="_blank" href="https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64">this link</a> if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.</p>

<p>(<em>For AWS</em>) If you'd like to train the agent on AWS (and have not <a target="_blank" href="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md">enabled a virtual screen</a>), then please use <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip">this link</a> (version 1) or <a target="_blank" href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip">this link</a> (version 2) to obtain the "headless" version of the environment.  You will <strong>not</strong> be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (<em>To watch the agent, you should follow the instructions to <a target="_blank" href="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md">enable a virtual screen</a>, and then download the environment for the <strong>Linux</strong> operating system above.</em>)</p>

Place the file in the DRLND GitHub repository, in the <code>p1_navigation/</code> folder, and unzip (or decompress) the file.

# Instructions
Follow the instructions in <code>DDPG/</code> and <code>DDPG_PER/</code> folders to get started with training the agents agent!

# Implementation Details
All implementation details and results are found in <code>Report/</code> folder

# References
<ul>
    <li>[1]: V. Mnih et al., "Human-level control through deep reinforcement learning", Nature, vol. 518, no. 7540, pp. 529-533, 2015. Available: 10.1038/nature14236 [Accessed 3 September 2021].
<li>[2]: U. Technologies, "Machine Learning Agents | Unity", Unity, 2021. [Online]. Available: https://unity.com/products/machine-learning-agents. [Accessed: 03- Sep- 2021]. 
<li>[3]: R. Sutton and A. Barto, Reinforcement Learning, 2nd ed. 2019.
<li>[4]: T. P. Lillicrap et al., "Continuous control with deep reinforcement learning", arXiv, vol. 150902971, 2015. [Accessed 4 October 2021].
<li>[5]: T. Schaul, J. Quan, I. Antonoglou and D. Silver, "Prioritized Experience Replay", arXiv, vol. 151105952, 2016. [Accessed 4 October 2021]. 
</ul>
