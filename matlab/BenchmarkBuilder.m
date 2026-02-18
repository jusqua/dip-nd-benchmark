classdef BenchmarkBuilder < handle
    properties (Access = private)
        specs
    end

    methods
        function obj = BenchmarkBuilder()
            obj.specs = {};
        end

        function attach(obj, name, func, varargin)
            post_func = [];
            if nargin > 3
                post_func = varargin{1};
            end

            spec = struct(...
                'name', name, ...
                'func', func, ...
                'post', post_func);

            obj.specs{end + 1} = spec;
        end

        function run(obj, rounds)
            if nargin < 2
                rounds = 1;
            end

            fprintf("operator,once");
            if rounds > 1
                fprintf(",mean");
            end
            fprintf("\n");

            for i = 1:length(obj.specs)
                spec = obj.specs{i};
                obj.perform_benchmark(rounds, spec);
            end
        end
    end

    methods (Access = private)
        function perform_benchmark(obj, rounds, spec)
            tic;
            spec.func();
            once_duration = toc;

            fprintf("%s,%f", spec.name, once_duration);

            if rounds <= 1
                fprintf("\n");
                if ~isempty(spec.post)
                    spec.post(spec.name);
                end
                return;
            end

            tic;
            for i = 1:rounds
                spec.func();
            end
            total_duration = toc;
            mean_duration = total_duration / rounds;

            fprintf(",%f\n", mean_duration);

            if ~isempty(spec.post)
                spec.post(spec.name);
            end
        end
    end
end
