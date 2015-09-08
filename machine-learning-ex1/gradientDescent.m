function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    
    %% Versao 1. Utilizando as variaveis t1 e t2 para armazenar theta temporario
    %  t1 = theta(1) - (alpha / m ) * sum((X * theta - y) .* X(:,1));
    %  t2 = theta(2) - (alpha / m ) * sum((X * theta - y) .* X(:,2));
    %  theta = [t1;t2]
     
    %% Versao 2. Depois percebi que poderia fazer tudo sem ficar usando os indices de theta. Utilizando vetorizacao
    % theta = theta - (alpha / m ) * sum((X * theta - y) .* X)';
    
    % Versao 3. A versao 2 tinha um problema em .* X, pois o octave um broadcast automatico na operacao. 
    % Ler mais aqui: https://www.gnu.org/software/octave/doc/interpreter/Broadcasting.html
    % Apos bravura, consegui reduzir ainda mais o calculo. Espero no fundo do meu coracao conseguir lembrar o que fiz
    % daqui a algumas semanas quando estiver olhando esse codigo novamente. 
    % 
    theta = theta - (alpha / m ) * (X'*(X * theta - y));
    
    
    % ============================================================

    % Save the cost J in every iteration    
    cost = computeCost(X, y, theta)
    J_history(iter) = cost;

end

end
