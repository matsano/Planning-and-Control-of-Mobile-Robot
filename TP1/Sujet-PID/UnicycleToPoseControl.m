function [ u ] = UnicycleToPoseControl( xTrue,xGoal )
%Computes a control to reach a pose for unicycle
%   xTrue is the robot current pose : [ x y theta ]'
%   xGoal is the goal point
%   u is the control : [v omega]'

Kp = 50;
Ka = 10;
Kb = 20;
alphaMax = 1;

error = xGoal - xTrue;
p = norm(error(1:2));
alpha = atan2(error(2), error(1)) - xTrue(3);
alpha = AngleWrap(alpha);

if p > 0.05
    w = Ka * alpha;
    if abs(alpha) > alphaMax
        v = 0;
    else
        v = Kp * p;
    end
else
    beta = error(3);
    w = Kb * beta;
    v = 0;
end

u = [v, w];
