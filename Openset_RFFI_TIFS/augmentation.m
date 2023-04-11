function [sig_out, myPathGain] = augmentation(sig_in, Ts)
    
    sig_len = length(sig_in);
    t_rms = ((300-10).*rand(1) + 10)*1e-9; % random RMS delay spread from 10 to 300 ns.
    [avgPathGains, pathDelays]= exp_PDP(t_rms, Ts);
    
    %                 wavelength = 3e8/868.1e6;
    %                 speed = (5-0).*rand(1) + 0;
    %                 fD = speed/wavelength;
    %                 fD = (10-0).*rand(1) + 0;
    
    fD = (10-0).*rand(1) + 0;
    k_factor = (10-0).*rand(1) + 0;
    
    wirelessChan = comm.RicianChannel('SampleRate',1/Ts,'KFactor',k_factor,'MaximumDopplerShift',fD,...
        'PathDelays',pathDelays,'AveragePathGains',avgPathGains,'DopplerSpectrum', doppler('Jakes'),...
        'PathGainsOutputPort',true);
    
    chanInfo = info(wirelessChan);
    delay = chanInfo.ChannelFilterDelay;
    
    chInput = [sig_in;zeros(50,1)];
    [chOut, myPathGain] = wirelessChan(chInput);
    sig_out = chOut(delay+1:sig_len+delay);
    
end

function [avgPathGains,pathDelays ]=exp_PDP(tau_d,Ts)

A_dB = -30;


sigma_tau = tau_d; 
A=10^(A_dB/10);
lmax=ceil(-tau_d*log(A)/Ts); % Eq.(2.2)

% Exponential PDP
p=0:lmax; 
pathDelays = p*Ts;


p = (1/sigma_tau)*exp(-p*Ts/sigma_tau);
p_norm = p/sum(p);


avgPathGains = 10*log10(p_norm); % convert to dB

end