classdef BenchmarkBuilder < handle
    properties (Access = private)
        specs
        gpuDev
    end

    methods
        function obj = BenchmarkBuilder(gpuDev)
            obj.specs = {};
            obj.gpuDev = gpuDev;
        end

        function attach(obj, name, type, group, func, varargin)
            post_func = [];
            if nargin > 5
                post_func = varargin{1};
            end

            spec = struct(...
                'name', name, ...
                'type', type, ...
                'group', group, ...
                'func', func, ...
                'post', post_func);

            obj.specs{end + 1} = spec;
        end

        function run(obj, rounds)
            if nargin < 2
                rounds = 1;
            end

            fprintf("operator,type,group,once");
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
            wait(obj.gpuDev);
            once_duration = toc;

            fprintf("%s,%s,%s,%f", spec.name, spec.type, spec.group, once_duration);

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
                wait(obj.gpuDev);
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
