	?S㥛????S㥛???!?S㥛???	???k7@???k7@!???k7@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?S㥛????&1???AH?z?G??Y??~j?t??*	     ?m@2U
Iterator::Model::ParallelMapV2!?rh????!?O??O?G@)!?rh????1?O??O?G@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatesh??|???!??o??oA@)??~j?t??17??7???@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???S㥛?!Cg?Bg?&@)???S㥛?1Cg?Bg?&@:Preprocessing2F
Iterator::ModelT㥛? ??!yxxxxxJ@)9??v????1?F??F?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!??????@)y?&1?|?1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??(\?µ?!?m۶m?A@)????Mb`?1??????:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 23.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t15.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???k7@IHz?%S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&1????&1???!?&1???      ??!       "      ??!       *      ??!       2	H?z?G??H?z?G??!H?z?G??:      ??!       B      ??!       J	??~j?t????~j?t??!??~j?t??R      ??!       Z	??~j?t????~j?t??!??~j?t??b      ??!       JCPU_ONLYY???k7@b qHz?%S@