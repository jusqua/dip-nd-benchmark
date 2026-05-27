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

            fprintf("operator,type,group,duration\n");
            for i = 1:length(obj.specs)
                spec = obj.specs{i};
                obj.perform_benchmark(rounds, spec);
            end
        end
    end

    methods (Access = private)
        function perform_benchmark(obj, rounds, spec)
            % Warm up
            spec.func();
            wait(obj.gpuDev);

            for i = 1:rounds
                tic;
                spec.func();
                wait(obj.gpuDev);
                duration = toc;

                fprintf("%s,%s,%s,%f\n", spec.name, spec.type, spec.group, duration);
            end

            if ~isempty(spec.post)
                spec.post(spec.name);
            end
        end
    end
end
