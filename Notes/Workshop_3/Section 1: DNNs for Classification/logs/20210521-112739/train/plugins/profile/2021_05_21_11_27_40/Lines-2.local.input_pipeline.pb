	=
ףp=??=
ףp=??!=
ףp=??	u?E]t)@u?E]t)@!u?E]t)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$=
ףp=??/?$????A;?O??n??Y??S㥛??*	     ?o@2U
Iterator::Model::ParallelMapV2?&1???!k?Dly(D@)?&1???1k?Dly(D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate#??~j???!??6@@)?Q?????1R	?;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????????!_ʘ?I?3@)J+???1S??N^3@:Preprocessing2F
Iterator::Modelh??|?5??!?????QG@)????Mb??1	??K@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?I+???!???,d@)?I+???1???,d@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??(\?µ?!??/e?@@)????Mbp?1	??K??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!	??K??)????MbP?1	??K??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t12.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9v?E]t)@I?E]t?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/?$????/?$????!/?$????      ??!       "      ??!       *      ??!       2	;?O??n??;?O??n??!;?O??n??:      ??!       B      ??!       J	??S㥛????S㥛??!??S㥛??R      ??!       Z	??S㥛????S㥛??!??S㥛??b      ??!       JCPU_ONLYYv?E]t)@b q?E]t?U@