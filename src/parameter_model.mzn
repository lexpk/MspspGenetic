int: nActs;
int: nResources;
int: nSkills;
int: nPrecs;

set of int: ACT = 1..nActs;
set of int: RESOURCE = 1..nResources;
set of int: SKILL = 1..nSkills;
set of int: PREC = 1..nPrecs;

array[ACT] of int: dur; % duration
array[ACT,SKILL] of int: sreq; % task's skill requirement
array[RESOURCE,SKILL] of bool: mastery; % resource's skill mastery
array[PREC] of ACT: pred; % predecessor activity
array[PREC] of ACT: succ; % successor activity

% REDUNDANT INFORMATION

int: mint;  % minumum makespan
int: maxt;  % maximum makespan

int: nUnrels; % number of unrelated activites wrt to the precedence graph
set of int: UNREL = 1..nUnrels;

array[UNREL] of ACT: unpred; % unrelated pair, first activity
array[UNREL] of ACT: unsucc; % unrelated pair, second activity

array[ACT] of set of RESOURCE: USEFUL_RES; % useful resources for each act
array[RESOURCE] of set of ACT: POTENTIAL_ACT; % potential activities for each res

