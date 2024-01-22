from numpy import savez, load 
from matplotlib.pyplot import style, figure, axes 
from celluloid import Camera 
from IPython.terminal.interactiveshell import TerminalInteractiveShell 
 
shell = TerminalInteractiveShell.instance() 
shell.define_macro('foo', """a,b=10,20""") 
 
results_of_calculations = load('results_of_calculations.npz') 
 
x = results_of_calculations['x'] 
u = results_of_calculations['u'] 
 
a = min(x); b = max(x) 
M = len(u) - 1 
 
from IPython import get_ipython 
get_ipython().run_line_magic('matplotlib', 'qt') 
 
style.use('dark_background') 
fig = figure() 
camera = Camera(fig) 
ax = axes(xlim=(a, b), ylim=(-2.0, 2.0)) 
ax.set_xlabel('x'); ax.set_ylabel('u') 
 
for m in range(0, M + 1, 30) : 
    ax.plot(x, u[m], color='y', ls='-', lw=2) 
    camera.snap() 
 
animation = camera.animate(interval=15, repeat=False, blit=True) 
animation.save('u.gif', dpi=100)