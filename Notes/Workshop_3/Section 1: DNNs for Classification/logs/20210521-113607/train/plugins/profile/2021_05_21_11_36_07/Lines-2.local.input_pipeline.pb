	T㥛? ??T㥛? ??!T㥛? ??	O??N??.@O??N??.@!O??N??.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$T㥛? ??+??????A?Q?????YV-???*	     ?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?&1???!5M?4ML@)????K7??1?,˲,?C@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??C?l???!? ??5@)???S㥻?1۶m۶m5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate?l??????!?u]?u]-@)?l??????1?u]?u]-@:Preprocessing2U
Iterator::Model::ParallelMapV2T㥛? ??!      )@)T㥛? ??1      )@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???S㥛?!۶m۶m@)Zd;?O???1AA@:Preprocessing2F
Iterator::Model??(\?µ?!?u]?u?0@)?I+???1]?u]?u@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?~j?t???!?0?0@)?~j?t???1?0?0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!Y?eY?e??)????Mbp?1Y?eY?e??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????MbP?!Y?eY?e??)????MbP?1Y?eY?e??:Preprocessing2a
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor:Preprocessing2b
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 15.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9P??N??.@Ivb'vb'U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+??????+??????!+??????      ??!       "      ??!       *      ??!       2	?Q??????Q?????!?Q?????:      ??!       B      ??!       J	V-???V-???!V-???R      ??!       Z	V-???V-???!V-???b      ??!       JCPU_ONLYYP??N??.@b qvb'vb'U@