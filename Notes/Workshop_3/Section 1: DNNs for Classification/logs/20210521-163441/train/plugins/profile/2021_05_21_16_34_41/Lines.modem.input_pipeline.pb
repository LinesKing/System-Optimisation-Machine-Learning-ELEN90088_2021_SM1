	??Q?????Q???!??Q???	????-4(@????-4(@!????-4(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??Q???sh??|???A??C?l??YR???Q??*	     ?f@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapX9??v???!8?v???@@)X9??v???18?v???@@:Preprocessing2U
Iterator::Model::ParallelMapV2V-???!??D???@)V-???1??D???@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!?(5?08@)???Q???1mާ?d0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1???!??Z9??@)y?&1???1??Z9??@:Preprocessing2F
Iterator::Model?V-??!/??fC@)9??v????1f?"Qj@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!???:
@)?~j?t?x?1???:
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9????-4(@Im?Kz?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	sh??|???sh??|???!sh??|???      ??!       "      ??!       *      ??!       2	??C?l????C?l??!??C?l??:      ??!       B      ??!       J	R???Q??R???Q??!R???Q??R      ??!       Z	R???Q??R???Q??!R???Q??b      ??!       JCPU_ONLYY????-4(@b qm?Kz?U@