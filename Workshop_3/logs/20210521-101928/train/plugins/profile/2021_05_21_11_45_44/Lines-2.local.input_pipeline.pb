	V-?????V-?????!V-?????	??׽?u@??׽?u@!??׽?u@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V-???????MbX??Affffff??YJ+???*	     ?R@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate????????!E>?S?@@)Zd;?O???1L?Ϻ??@:Preprocessing2U
Iterator::Model::ParallelMapV2??~j?t??!Ϻ???9@)??~j?t??1Ϻ???9@:Preprocessing2F
Iterator::Modely?&1???!?n0E>?B@);?O??n??11E>?S(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9??v????!??L?1@);?O??n??11E>?S(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipZd;?O???!L?Ϻ?O@){?G?zt?1o0E>?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!?Y7?"?@)????Mbp?1?Y7?"?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy?&1???!?n0E>?B@)?~j?t?h?1v?)?Y7@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice????MbP?!?Y7?"???)????MbP?1?Y7?"???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor????MbP?!?Y7?"???)????MbP?1?Y7?"???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t15.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??׽?u@IP?"?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??MbX????MbX??!??MbX??      ??!       "      ??!       *      ??!       2	ffffff??ffffff??!ffffff??:      ??!       B      ??!       J	J+???J+???!J+???R      ??!       Z	J+???J+???!J+???b      ??!       JCPU_ONLYY??׽?u@b qP?"?W@