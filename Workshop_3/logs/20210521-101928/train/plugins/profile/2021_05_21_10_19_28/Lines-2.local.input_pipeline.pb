	??Q?????Q???!??Q???	wc?#r?4@wc?#r?4@!wc?#r?4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??Q????V-??AV-???Y??S㥛??*	      n@2U
Iterator::Model::ParallelMapV2?A`??"??!UUUUUF@)?A`??"??1UUUUUF@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj?t???!??????A@)j?t???1??????A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9??v????!??????%@)?I+???1UUUUUU"@:Preprocessing2F
Iterator::Model???Q???!      I@)y?&1???1UUUUUU@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mb??!??????
@)????Mb??1??????
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!????????)????Mbp?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 20.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s9.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9wc?#r?4@I"'wc?S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V-???V-??!?V-??      ??!       "      ??!       *      ??!       2	V-???V-???!V-???:      ??!       B      ??!       J	??S㥛????S㥛??!??S㥛??R      ??!       Z	??S㥛????S㥛??!??S㥛??b      ??!       JCPU_ONLYYwc?#r?4@b q"'wc?S@