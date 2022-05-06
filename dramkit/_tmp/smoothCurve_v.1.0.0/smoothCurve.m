function [smoothY]=smoothCurve(Y,varargin)
%   ONE DIMENSIONAL SMOOTHING FUNCTION 
%
%   smoothCurve offers various different window smoothing options.
%   Default window function is Konno-Ohmachi (see Konno and Ohmachi (1998),
%   page 234), which is symmetric in log space. This function uses
%   convolution method to filter the signal for smoothing. 
%
%   USAGE:
%
%   [smoothY] = smoothCurve(Y,varargin)
%
%   STATIC INPUT:
%
%             Y = input signal (1xn or nx1)
%
%   VALID PROP_NAME / PROP_VAL PAIRS:
%   -----------------------------------------
%   'w'       --> (1x1)-[numeric]-[default:40]
%   'b'       --> (1x1)-[numeric]-[default: 20]
%   'method'  --> [text]-[default: konno-ohmachi]
%   'debug'   --> [text]-[default: False]
%
%    NOTES:
%             w = width of window function (e.g., 100)
%
%             b = bandwidth coefficient of konno-ohmachi window (e.g., 20)
%
%        method = window function (e.g., boxcar, gaussian, hamming, hann,
%                 hanning, konno-ohmachi, parzen, triang)
%
%         debug = 'True' for print debug messages
%
%   OUTPUT:
%
%    smoothY = smoothed array (1xn)
%
%   EXAMPLES: 
%
%   see demo.m file
%
%   REQUIREMENTS:
%
%   smoothCurve function does not require any MatLAB toolbox
%
%   ACKNOWLEDGEMENT:
%
%   In preparing this function, I benefitted from konv.m function written
%   by Ali Jadoon, which is available at MathWorks FEX.
%
%   REFERENCE:
%
%   Konno, K. and Ohmachi, T. (1998) "Ground-motion characteristics
%   estimated from spectral ratio between horizontal and vertical
%   components of microtremor," Bulletin of the Seismological Society of
%   America, 88(1): 228-241.
%
%   THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
%   WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
%   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
%   NO EVENT SHALL THE COPYRIGHT OWNER BE LIABLE FOR ANY DIRECT, INDIRECT,
%   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
%   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
%   OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
%   TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
%   USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
%   DAMAGE.
%
%   Written by Dr. Erol Kalkan, P.E. (kalkan76@gmail.com)
%   $Revision: 1.0.0 $  $Date: 2019/02/26 12:00:00 $
%
%% DEFAULT PROPERTIES
if (nargin >= 1)
    w = 40;
    b = 20;
    method = 'konno-ohmachi';
    debug = 'False';
else
    error('smoothCurve: First argument must be a waveform')
end

%% USER-DEFINED PROPERTIES
if (nargin > 1)
    v = varargin;
    nv = nargin-1;
    if ~rem(nv,2) == 0
        error(['smoothCurve: Arguments after time series must ',...
            'appear in property name/val pairs'])
    end
    for n = 1:2:nv-1
        name = lower(v{n});
        val = v{n+1};
        switch name
            case 'w'
                w = val;
            case 'debug'
                debug = val;
            case 'b'
                b = val;
            case 'method'
                method = val;
            otherwise
                error('smoothCurve: Property name not recognized')
        end
    end
end

%% DEBUG OPTION
if strcmp(debug,'True')
        fprintf('applied window .....%s\n', method);
        fprintf('window width .......%d\n', w);
    if strcmp(method, 'konno-ohmachi')
        fprintf('b value ............%d\n', b);
    end
end

%% MAIN
% enforce window width as odd number
if mod(w,2) == 0
    w = w - 1;
end

halfw = round(w/2);

switch lower(method)
    case 'boxcar'
        W = boxcar(w);
    case 'gaussian'
        W = gausswin(w);
    case 'hamming'
        W = hamming(w);
    case 'hann'
        W = hann(w);
    case 'hanning'
        W = hanning(w);
    case 'konno-ohmachi'
        W = (sin(b * log10((1:w)/halfw))./(b*log10((1:w)/halfw))).^ 4;
        % enforce window function to be 1 at central value, which is halfw
        W(halfw) = 1;
    case 'parzen'
        W = parzenwin(w);
    case 'triang'
        W = triang(w);
    otherwise
        disp('smoothY: ''method'' is not recognized')
end

% enforce input as row vector
if ~isrow(Y); Y = Y'; end

% enforce weighting function as row vector
if ~isrow(W); W = W'; end

% convolution of signal (Y) with window function (h), which gives
% length(arg) = length(Y) + length(h) - 1

% normalize window function
h = W/sum(W);

N = length(Y);
d = w-1;

% pre-allocation of convolved signal
arg = zeros(1,N+d);

% padding
Y_pad = zeros(1,2*d+N);

% allocating Y_pad with input Y
Y_pad(w:end-d) = Y;

% flipping normalized window
hf = fliplr(h);

% shifting and multiplication
for i = 1:N+d
    arg(i)=sum(hf.*Y_pad(i:i+d));
end

% extract central values of convolved signal only so that smoothY has same
% length as input signal Y
smoothY = arg(halfw:end-halfw+1);
end